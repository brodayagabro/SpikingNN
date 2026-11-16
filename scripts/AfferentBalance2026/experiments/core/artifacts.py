"""Portable solutions, explicit legacy adapters and atomic experiment artifacts.

Only this boundary dispatches persisted schemas. Numerical code consumes typed
values, never filenames or guesses based on chromosome length.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import importlib.metadata
import inspect
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np

from .config import MODEL, content_hash, project_root, read_json
from .genome import GenomeConfig, NAMES, SCHEMA as CONTROLLER_SCHEMA, file_hash, load_genome, write_json

SOLUTION_SCHEMA = 'afferent_solution_v1'
POPULATION_SCHEMA = 'afferent_population_v1'
RUN_SCHEMA = 'afferent_result_v1'
POINTER_SCHEMA = 'afferent_solution_pointer_v1'


@dataclass
class Solution:
    """An interpreted controller, its command law and the original saved protocol."""
    config: GenomeConfig
    physical: np.ndarray
    command: np.ndarray | None = None
    kind: str = 'none'
    degree: int = 0
    current_tau_s: float = 0.
    effective_config: dict | None = None
    metadata: dict = field(default_factory=dict)
    path: Path | None = None
    sha256: str | None = None

    @property
    def codec(self):
        """Build the codec named by the validated document, never by array length."""
        return make_codec(self.config, self.kind, self.degree)

    @property
    def unit(self):
        """Encode physical values in this solution's own domain."""
        return encode(self.codec, self.physical, self.command)

    def describe(self):
        """Summarize source, model, dimensions and physiological times for notebooks."""
        fields = self.config.data['parameters']
        return dict(path=None if self.path is None else str(self.path), sha256=self.sha256,
                    model=MODEL, dt_ms=.1, experiment=self.config.data['experiment'],
                    physical_fields=len(getattr(self.codec, 'names', NAMES)),
                    independent_coordinates=self.codec.size, mirror_symmetry=self.config.mirror_symmetry,
                    muscle_times_ms={f['name']:float(self.physical[f['index']]) for f in fields
                                     if f['name'].endswith(('.tau_c', '.tau_1'))},
                    physiological_domain={f['name']:{k:f[k] for k in ('mutable', 'minimum', 'maximum') if k in f}
                                          for f in fields if f['name'].startswith('muscles.') and
                                          f['name'].endswith(('.tau_c', '.tau_1', '.A', '.w'))},
                    selection=self.metadata.get('selection', self.metadata))


def make_codec(config, kind=None, degree=3):
    """Adapt the four current tasks to the common executor's encode/decode contract."""
    experiment = config.data['experiment']
    if experiment in ('A_balance', 'B_balance'):
        return config
    if experiment == 'C_setpoint_balance':
        from ..C_setpoint_balance.joint_study import JointGenome
        return JointGenome(config, kind or 'polynomial', degree)
    if experiment == 'D_trajectory':
        from ..D_trajectory.controller import TrajectoryGenome
        return TrajectoryGenome(config)
    raise ValueError(f'Unsupported task schema: {experiment}')


def controller(codec):
    """Return the physical domain independently of command representation."""
    return codec if isinstance(codec, GenomeConfig) else codec.controller


def decode(codec, unit):
    """Return (physical controller, optional command matrix) for any current task."""
    return (codec.decode(unit), None) if isinstance(codec, GenomeConfig) else codec.decode(unit)


def encode(codec, physical, command=None):
    """Encode a physical candidate without copying foreign normalized coordinates."""
    return codec.encode(physical) if isinstance(codec, GenomeConfig) else codec.encode(physical, command)


def genome_payload(codec, unit, tau=0., **metadata):
    """Produce an embedded, independently readable historical genome document."""
    if isinstance(codec, GenomeConfig):
        return codec.payload(codec.decode(unit), **metadata)
    if controller(codec).data['experiment'] == 'D_trajectory':
        return codec.payload(unit, current_tau_s=tau, **metadata)
    return codec.payload(unit, **metadata)


def save_solution(path, codec, unit, effective_config, *, tau=0., selection=None, provenance=None):
    """Save a self-contained result; it remains readable after copying this file alone."""
    payload = genome_payload(codec, unit, tau, model=MODEL, selection=selection or {})
    document = dict(schema=SOLUTION_SCHEMA, model=MODEL, genome=payload,
                    effective_config=deepcopy(effective_config), selection=selection or {},
                    provenance=provenance or {})
    document['genome_sha256'] = content_hash(payload)
    write_json(path, document)
    return document


def load_solution(source, *, rank=1, runtime='current'):
    """Read versioned JSON or supported NPZ, retaining model/domain/provenance.

    Current replay is explicit. Archived runtime execution belongs to the
    historical runtime tool and is never silently mixed into this process.
    """
    if runtime != 'current':
        raise ValueError('Historical runtime requires an isolated source snapshot; use tools/runtime_support.py explicitly')
    path = None if isinstance(source, dict) else Path(source).expanduser().resolve()
    if path is not None and path.is_dir():
        path = path/'solution.json' if (path/'solution.json').is_file() else path/'best_genome.json'
    if path is not None and path.suffix == '.npz':
        return load_population(path, rank=rank)
    data = deepcopy(source) if isinstance(source, dict) else read_json(path)
    schema = data.get('schema')
    if schema == POINTER_SCHEMA:
        if path is None:
            raise ValueError('A relative solution pointer requires its source filename')
        target = (path.parent/data['path']).resolve()
        if file_hash(target) != data['sha256']:
            raise ValueError('Current-solution pointer checksum mismatch')
        return load_solution(target, rank=rank)
    effective, meta = None, data.get('metadata', {})
    if schema == SOLUTION_SCHEMA:
        if data['model'] != MODEL or content_hash(data['genome']) != data['genome_sha256']:
            raise ValueError('Portable solution model or genome checksum mismatch')
        effective, meta = data['effective_config'], {'selection': data['selection'], 'provenance': data['provenance']}
        data = data['genome']
        schema = data['schema']
    if schema == CONTROLLER_SCHEMA:
        cfg, p = load_genome(data)
        solution = Solution(cfg, p)
    elif schema == 'afferent_setpoint_joint_v1':
        from ..C_setpoint_balance.joint_study import load_joint_genome
        codec, unit = load_joint_genome(data)
        p, command = codec.decode(unit)
        solution = Solution(codec.controller, p, command, codec.kind, codec.degree)
    elif schema == 'afferent_trajectory_v1':
        from ..D_trajectory.controller import load_solution as read_trajectory
        cfg, p, command, tau, _ = read_trajectory(data)
        solution = Solution(cfg, p, command, 'polynomial', 3, tau)
    else:
        raise ValueError(f'Unsupported genome schema {schema!r}; historical current slots 46/47 '
                         'require the explicit import_legacy converter')
    solution.effective_config, solution.metadata = effective, meta
    solution.path = path
    solution.sha256 = file_hash(path) if path else content_hash(data)
    return solution


def load_population(path, rank=1):
    """Read a ranked new/legacy population using explicit schema and saved domain."""
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as data:
        schema = str(data['schema']) if 'schema' in data else None
        if schema == POPULATION_SCHEMA:
            template = json.loads(str(data['solution_template']))
            base = load_solution(template)
            codec = base.codec
        else:
            cfg_path = next((p/'genome_cfg.json' for p in path.parents if (p/'genome_cfg.json').is_file()), None)
            if cfg_path is None:
                raise ValueError('Legacy NPZ requires its saved genome_cfg.json; arrays alone are ambiguous')
            cfg = GenomeConfig(cfg_path)
            if schema not in (CONTROLLER_SCHEMA, 'afferent_setpoint_joint_v1', 'afferent_trajectory_v1', None):
                raise ValueError(f'Unsupported population schema {schema!r}')
            if schema is None and cfg.data['experiment'] != 'D_trajectory':
                raise ValueError('Missing population schema; use an explicit historical converter')
            if cfg.data['experiment'] == 'C_setpoint_balance':
                record_path = next((p/'run.json' for p in path.parents if (p/'run.json').is_file()), None)
                if record_path is None:
                    raise ValueError('Legacy C NPZ requires its saved command specification')
                spec = read_json(record_path)['command']
                codec = make_codec(cfg, spec['kind'], spec['degree'])
            else:
                codec = make_codec(cfg)
            if 'configuration_sha256' in data and str(data['configuration_sha256']) != cfg.signature:
                raise ValueError('Legacy population configuration checksum mismatch')
            base = None
            if cfg.data['experiment'] == 'D_trajectory':
                genome_path = next((parent/name for parent in path.parents
                                    for name in ('initial_genome.json', 'best_genome.json')
                                    if (parent/name).is_file()), None)
                if genome_path is None:
                    raise ValueError('Legacy D NPZ requires a saved initial/best genome with current_tau_s')
                base = load_solution(genome_path)
        score_key = next((key for key in ('validation_scores', 'training_scores', 'scores') if key in data), None)
        if score_key is None:
            raise ValueError('Population has no explicit ranking scores')
        if type(rank) is not int or not 1 <= rank <= len(data[score_key]):
            raise ValueError('Rank is outside the population')
        index = int(np.argsort(-data[score_key], kind='stable')[rank-1])
        unit = data['normalized' if 'normalized' in data else 'population'][index].copy()
        p, command = decode(codec, unit)
        if 'physical' in data:
            actual = p if command is None else np.r_[p, command.ravel()]
            np.testing.assert_allclose(actual, data['physical'][index], rtol=0, atol=1e-11)
        if 'names' in data and tuple(data['names']) != tuple(getattr(codec, 'names', NAMES)):
            raise ValueError('Population names do not match the declared schema')
        solution = load_solution(genome_payload(codec, unit, base.current_tau_s if base else 0.))
        solution.effective_config = base.effective_config if base else None
        solution.metadata = dict(selection=score_key, population_index=index, rank=rank,
                                 generation=int(data['generation']) if 'generation' in data else None)
        solution.path, solution.sha256 = path, file_hash(path)
        return solution


def transfer(solution, codec, *, allow_mechanics=False):
    """Validate every destination field and re-encode physical values with named errors."""
    cfg = controller(codec)
    errors = []
    if not allow_mechanics and not np.array_equal(solution.config.mechanics, cfg.mechanics):
        errors.append(f'model.mechanics: source {solution.config.mechanics.tolist()} differs from destination {cfg.mechanics.tolist()}')
    for field, value in zip(cfg.data['parameters'], solution.physical):
        if not field['mutable'] and value != field['value']:
            errors.append(f'{field["name"]}: {value:g}; expected fixed {field["value"]:g}')
        elif field['mutable'] and not field['minimum'] <= value <= field['maximum']:
            errors.append(f'{field["name"]}: {value:g}; expected [{field["minimum"]:g}, {field["maximum"]:g}]')
    if errors:
        raise ValueError('Incompatible source solution:\n'+'\n'.join(errors))
    command = solution.command
    if not isinstance(codec, GenomeConfig):
        width = codec.width if hasattr(codec, 'width') else 4
        if command is None:
            command = np.zeros((2, width))
        elif solution.kind == 'table' and getattr(codec, 'kind', 'polynomial') != 'table':
            raise ValueError('A current table needs an explicit polynomial fitting stage')
        elif command.shape[1] != width:
            if solution.kind == 'polynomial' and command.shape[1] < width:
                command = np.pad(command, ((0, 0), (0, width-command.shape[1])))
            elif solution.kind == 'polynomial' and np.all(command[:, width:] == 0):
                command = command[:, :width]
            else:
                raise ValueError('Command degree differs; explicit current-law conversion is required')
    elif command is not None and np.any(command):
        raise ValueError('A/B require zero applied currents; command removal must be explicit')
    return encode(codec, solution.physical, command)


def atomic_npz(path, **arrays):
    """Write compressed arrays atomically without changing established array values."""
    path = Path(path)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def environment_versions():
    """Record the interpreter and installed dependency versions."""
    result = {'python': sys.version}
    for name in ('SpikingNN', 'numpy', 'numba', 'scipy', 'matplotlib'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = 'unavailable distribution metadata'
    return result


def source_snapshot(output=None):
    """Archive local calculation/UI handlers and available library Python sources."""
    import SpikingNN.core.Izh_net as library
    root = project_root()
    library_root = Path(inspect.getfile(library)).parent.parent
    files = {'experiment/'+str(p.relative_to(root)):p for p in root.rglob('*.py')
             if not {'results', '__pycache__', 'archive', 'pdf_build'}.intersection(p.relative_to(root).parts)}
    files.update({'experiment/'+str(p.relative_to(root)):p for p in (root/'configs').rglob('*.json')})
    files.update({'library/SpikingNN/'+str(p.relative_to(library_root)):p for p in library_root.rglob('*.py')})
    hashes = {name:file_hash(path) for name, path in sorted(files.items())}
    if output is None:
        return hashes
    with zipfile.ZipFile(Path(output)/'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
    return hashes


def publish(solution_path, pointer):
    """Explicitly publish a pointer; never alter the source solution or its run."""
    import os
    solution = load_solution(solution_path)
    pointer = Path(pointer).resolve()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    record = dict(schema=POINTER_SCHEMA, path=os.path.relpath(solution.path, pointer.parent),
                  sha256=solution.sha256, model=MODEL, domain_sha256=solution.config.signature,
                  selection=solution.metadata)
    write_json(pointer, record)
    return record


def prepare_solution(effective, *, dry_run=False):
    """Apply only explicitly requested changes and save their physical-value audit."""
    from .config import controller_config
    from .genome import project_mirror
    initial = effective['initialization']
    if not initial['solution']:
        raise ValueError('prepare requires --solution')
    source = load_solution(initial['solution'])
    cfg = controller_config(effective)
    codec = make_codec(cfg, source.kind if source.kind != 'none' else None,
                       effective['genome']['command'].get('degree', 3))
    prepared = initial['prepare']
    physical = source.physical.copy()
    command = None if source.command is None else source.command.copy()
    for name in prepared['clip_fields']:
        field = cfg.data['parameters'][NAMES.index(name)]
        if not field['mutable']:
            raise ValueError(f'{name}: clipping a fixed field is ambiguous; use prepare.values')
        physical[field['index']] = np.clip(physical[field['index']], field['minimum'], field['maximum'])
    for name, value in prepared['values'].items():
        physical[NAMES.index(name)] = value
    if prepared['gain_match']:
        for side in ('flexor', 'extensor'):
            def position(key):
                return NAMES.index(f'muscles.{side}.{key}')
            w, n, tc, t1 = [position(key) for key in ('w', 'N', 'tau_c', 'tau_1')]
            weight = source.physical[w]*source.physical[tc]/physical[tc]
            for index, value in ((w, weight), (n, source.physical[n]*source.physical[t1]/physical[t1])):
                field = cfg.data['parameters'][index]
                physical[index] = np.clip(value, field['minimum'], field['maximum']) if field['mutable'] else field['value']
            row = 0 if side == 'flexor' else 1
            for neuron in range(2):
                physical[NAMES.index(f'output.P.{row}.{neuron}')] *= weight/physical[w]
    if prepared['mirror_projection']:
        if not cfg.mirror_symmetry:
            raise ValueError('prepare.mirror_projection requires genome.mirror_symmetry=true')
        physical = project_mirror(physical)
        if command is not None:
            reflection = (-1.)**np.arange(command.shape[1]) if source.kind == 'polynomial' else None
            first = .5*(command[0]+(command[1]*reflection if reflection is not None else command[1, ::-1]))
            command = np.array([first, first*reflection if reflection is not None else first[::-1]])
    transformed = Solution(cfg, physical, command, source.kind, source.degree, source.current_tau_s)
    if not np.array_equal(source.config.mechanics, cfg.mechanics):
        raise ValueError('prepare does not silently change model.mechanics')
    unit = transfer(transformed, codec)
    changes = [dict(name=NAMES[i], before=float(source.physical[i]), after=float(physical[i]))
               for i in np.flatnonzero(physical != source.physical)]
    record = dict(source=str(source.path), source_sha256=source.sha256, operations=prepared,
                  changes=changes, command_changed=not np.array_equal(source.command, command),
                  note='Projection and steady-gain matching do not preserve transient dynamics or performance.')
    if dry_run:
        return record
    output = Path(effective['storage']['output'])
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema=RUN_SCHEMA, status='running', action='prepare', model=MODEL,
                    versions=environment_versions(), source_solution_sha256=source.sha256)
    write_json(output/'manifest.json', manifest)
    try:
        (output/'inputs').mkdir()
        shutil.copy2(source.path, output/'inputs'/source.path.name)
        write_json(output/'effective_config.json', effective)
        save_solution(output/'solution.json', codec, unit, effective, tau=source.current_tau_s,
                      selection={'role':'prepared_initialization', 'test_used_for_selection':False}, provenance=record)
        write_json(output/'preparation.json', record)
        manifest.update(status='completed', source_sha256=source_snapshot(output),
                        solution_sha256=file_hash(output/'solution.json'),
                        artifacts={'solution':'solution.json', 'configuration':'effective_config.json'})
    except BaseException as error:
        manifest.update(status='failed', error=dict(type=type(error).__name__, message=str(error)))
        write_json(output/'manifest.json', manifest)
        raise
    write_json(output/'manifest.json', manifest)
    return dict(status='completed', output=str(output), solution=str(output/'solution.json'), changes=changes)


def compare_runs(runs, output):
    """Compare existing metrics in separate protocol groups without any simulation."""
    from .run import write_csv
    rows = []
    for run in runs:
        path = Path(run).resolve()
        if path.is_file():
            path = path.parent
        manifest = read_json(path/'manifest.json')
        if manifest.get('schema') != RUN_SCHEMA or manifest['status'] != 'completed':
            raise ValueError(f'{path}: expected a completed versioned result')
        solution = load_solution(path/'solution.json')
        cfg = solution.effective_config
        if file_hash(path/'solution.json') != manifest['solution_sha256']:
            raise ValueError(f'{path}: selected solution differs from its manifest')
        bank_files = sorted(p for p in (path/'banks').glob('*') if p.is_file())
        group = content_hash(dict(model=manifest['model'], experiment=cfg['experiment'],
                                  mechanics=cfg['model']['mechanics'], criterion=cfg['protocol']['criterion'],
                                  banks={p.name:file_hash(p) for p in bank_files}))
        row = dict(run=str(path), group=group, experiment=cfg['experiment'],
                   solution_sha256=solution.sha256, selected_generation=solution.metadata.get('selection', {}).get('generation'),
                   validation_fitness=None, test_fitness=None, test_successes=None)
        for name in ('validation', 'test'):
            archive = path/'metrics'/f'{name}.npz'
            if archive.exists():
                with np.load(archive, allow_pickle=False) as data:
                    row[name+'_fitness'] = float(data['scores'][0])
                    if name == 'test':
                        row['test_successes'] = int(data['metrics'][:, 5].sum())
        rows.append(row)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output/'comparison.csv', rows)
    groups = sorted({r['group'] for r in rows})
    write_json(output/'comparison.json', dict(groups=groups, runs=rows, missing_test_is_not_computed=True,
                                               ranking='Compare values only within the same group.'))
    return dict(status='completed', output=str(output), groups=len(groups), runs=len(rows))
