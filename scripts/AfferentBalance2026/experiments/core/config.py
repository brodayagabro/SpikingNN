"""Resolve local experiment presets, named overrides and explicit input paths.

This module performs configuration inspection, not simulation. The controller
schema and the run schema have separate versions. Presets have one inheritance
level; an effective configuration contains its entire resolved domain.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .genome import GenomeConfig, NAMES, SCHEMA as GENOME_SCHEMA

SCHEMA = 'afferent_experiment_run_v1'
MODEL = 'SpikingNN.Afferents/v3-tensile'
EXPERIMENT_NAMES = dict(A='A_balance', B='B_balance', C='C_setpoint_balance', D='D_trajectory')
POLICIES = {
    'final_validation_v1', 'checkpoint_validation_v1',
    'initial_checkpoint_validation_v1', 'd_comparison_validation_v1',
}
BLOCK_KEYS = {
    'model': {'name', 'dt_ms', 'architecture', 'mechanics'},
    'genome': {'schema', 'parameters', 'overrides', 'mirror_symmetry', 'command'},
    'protocol': {'criterion', 'selection', 'balance', 'setpoint', 'trajectory', 'replay', 'evaluate_test'},
    'initialization': {'solution', 'extra_solutions', 'sources', 'strategy', 'local_sigma',
                       'candidate_seed_offset', 'current_tau_s', 'prepare'},
    'search': {'method', 'population', 'generations', 'seed', 'mutation_probability',
               'sigma_start', 'sigma_end', 'validation_every', 'checkpoints',
               'table_generations', 'polarity_seeds'},
    'sweep': {'axes'},
    'runtime': {'workers'},
    'storage': {'output', 'save_generations', 'replay', 'recording_stride', 'figures'},
}
NESTED_KEYS = {
    'model.mechanics': {'mass_kg', 'length_m', 'viscosity', 'gravity_m_s2'},
    'genome.command': {'degree', 'coefficient_bound_nA', 'table_current_bound_nA'},
    'protocol.balance': {'duration_s', 'training_count', 'validation_count', 'test_count',
                         'bank_seeds', 'initial_conditions'},
    'protocol.balance.bank_seeds': {'training', 'validation', 'test'},
    'protocol.balance.initial_conditions': {'angle_range_deg', 'speed_range_rad_s'},
    'protocol.setpoint': {'targets_deg', 'validation_targets_deg', 'test_targets_deg'},
    'protocol.trajectory': {'hold_s', 'ramp_s', 'training_count', 'validation_count', 'test_count',
                            'bank_seeds', 'filter_times_s', 'duration_s'},
    'protocol.trajectory.bank_seeds': {'training', 'validation', 'test'},
    'protocol.replay': {'angles_deg', 'speeds_rad_s', 'targets_deg', 'duration_s'},
    'initialization.prepare': {'clip_fields', 'values', 'mirror_projection', 'gain_match'},
    'sweep.axes': {'population', 'generations', 'seed', 'mutation_probability', 'method', 'degree'},
}
FIELD_KEYS = {'value', 'mutable', 'minimum', 'maximum', 'scale'}


def project_root():
    """Find the local experiment by its files, without a fixed parent-depth rule."""
    for directory in Path(__file__).resolve().parents:
        if (directory/'TZ.md').is_file() and (directory/'experiments/core/genome.py').is_file():
            return directory
        if (directory/'run.py').is_file() and (directory/'configs/presets').is_dir():
            return directory
    raise FileNotFoundError('Cannot locate the AfferentBalance2026 experiment')


def repository_root():
    """Locate the repository for legacy provenance, independently of directory names."""
    for directory in project_root().parents:
        if (directory/'src/SpikingNN').is_dir():
            return directory
    raise FileNotFoundError('Cannot locate the SpikingNN repository')


def read_json(path):
    """Reject nonstandard JSON NaN/Infinity values before applying overrides."""
    def reject(value):
        raise ValueError(f'Nonfinite JSON number: {value}')
    return json.loads(Path(path).read_text(), parse_constant=reject)


def content_hash(value):
    """Hash a finite canonical JSON value."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _check_keys(value, allowed, location):
    """Reject misspellings with the full configuration path."""
    if not isinstance(value, dict):
        raise ValueError(f'{location}: expected an object')
    unknown = set(value)-allowed
    if unknown:
        raise ValueError(f'{location}: unsupported keys {sorted(unknown)}; supported: {sorted(allowed)}')


def _validate_keys(data, *, full=False):
    """Check key applicability before merging, including nested overrides."""
    allowed = {'schema', 'preset', 'experiment', *BLOCK_KEYS}
    if full:
        allowed |= {'provenance', 'controller', 'inspection', 'action'}
    _check_keys(data, allowed, 'config')
    for name, allowed in BLOCK_KEYS.items():
        if name in data:
            _check_keys(data[name], allowed, name)
    for dotted, allowed in NESTED_KEYS.items():
        node = data
        for part in dotted.split('.'):
            if not isinstance(node, dict) or part not in node:
                break
            node = node[part]
        else:
            _check_keys(node, allowed, dotted)
    gene_overrides = data.get('genome', {}).get('overrides', {})
    _check_keys(gene_overrides, set(NAMES), 'genome.overrides')
    for name, values in gene_overrides.items():
        if name not in NAMES:
            raise ValueError(f'genome.overrides.{name}: unknown field; adding neurons/receptors '
                             'requires a new controller schema and numerical implementation')
        _check_keys(values, FIELD_KEYS, f'genome.overrides.{name}')
    prepared = data.get('initialization', {}).get('prepare', {})
    for name in [*prepared.get('clip_fields', []), *prepared.get('values', {})]:
        if name not in NAMES:
            raise ValueError(f'initialization.prepare: unknown controller field {name}')


def _leaves(value, prefix=''):
    """Yield leaf paths; arrays are atomic configuration values."""
    if isinstance(value, dict) and value:
        for key, child in value.items():
            yield from _leaves(child, f'{prefix}.{key}' if prefix else key)
    else:
        yield prefix, value


def _merge(target, update, origin, sources, prefix=''):
    """Merge one explicit layer and preserve the provenance of each leaf."""
    for key, value in update.items():
        dotted = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value, origin, sources, dotted)
        else:
            target[key] = deepcopy(value)
            for leaf, _ in _leaves(value, dotted):
                sources[leaf] = origin


def _resolve_paths(data, base, paths, origin):
    """Anchor JSON paths to the declaring file, not the process working directory."""
    for block, key in [('storage', 'output'), ('initialization', 'solution')]:
        raw = data.get(block, {}).get(key)
        if raw is not None:
            p = Path(raw).expanduser()
            resolved = (p if p.is_absolute() else base/p).resolve()
            data[block][key] = str(resolved)
            paths[f'{block}.{key}'] = dict(original=str(raw), resolved=str(resolved), source=origin)
    for key in ('extra_solutions', 'sources'):
        if key in data.get('initialization', {}):
            raw = data['initialization'][key]
            if not isinstance(raw, list) or any(not isinstance(p, str) for p in raw):
                raise ValueError(f'initialization.{key}: expected a list of paths')
            values = [str((base/Path(p).expanduser()).resolve()) for p in raw]
            data['initialization'][key] = values
            paths[f'initialization.{key}'] = dict(original=raw, resolved=values, source=origin)


def legacy_config(data, source=None):
    """Adapt the old controller configuration explicitly, recording ignored fields."""
    _check_keys(data, {'schema', 'experiment', 'dt_ms', 'parameters', 'mechanics', 'simulation',
                       'setpoint', 'trajectory', 'ga', 'sweep', 'mirror_symmetry'}, 'legacy config')
    for block, keys in (
        ('ga', {'population', 'generations', 'seed', 'mutation_probability', 'workers',
                'sigma_start', 'sigma_end', 'validation_every'}),
        ('simulation', NESTED_KEYS['protocol.balance'] | {'replay_angles_deg', 'save_top'}),
        ('setpoint', NESTED_KEYS['protocol.setpoint'] | NESTED_KEYS['genome.command'] |
                     {'checkpoints', 'training_enabled', 'seed_genome', 'currents', 'save_figures'}),
        ('trajectory', NESTED_KEYS['protocol.trajectory'] | {'coefficient_bound_nA', 'ga', 'description'}),
        ('sweep', {'populations', 'generations', 'mutation_probabilities', 'seeds', 'degrees', 'methods'}),
    ):
        if block in data:
            _check_keys(data[block], keys, f'legacy.{block}')
    kind = data['experiment'][0]
    preset = read_json(project_root()/'configs/presets'/f'{EXPERIMENT_NAMES[kind]}_tau50_v1.json')
    preset['genome']['parameters'] = deepcopy(data['parameters'])
    preset['genome']['mirror_symmetry'] = data.get('mirror_symmetry', False)
    preset['model'].update(dt_ms=data['dt_ms'], mechanics=deepcopy(data['mechanics']))
    messages = []
    if 'simulation' in data:
        simulation = deepcopy(data['simulation'])
        preset['protocol']['replay']['angles_deg'] = simulation.pop('replay_angles_deg', [0.])
        if 'save_top' in simulation:
            simulation.pop('save_top')
            messages.append('simulation.save_top: legacy top-N replay is not a new-run selection policy; only the selected solution is replayed')
        preset['protocol']['balance'] = simulation
    if kind == 'C':
        old = data['setpoint']
        preset['protocol']['setpoint'] = {k: deepcopy(old[k]) for k in NESTED_KEYS['protocol.setpoint']}
        preset['genome']['command'].update({k: old[k] for k in ('degree', 'coefficient_bound_nA', 'table_current_bound_nA')})
        preset['search']['checkpoints'] = old['checkpoints']
        for key in ('training_enabled', 'seed_genome', 'currents', 'save_figures'):
            if key in old:
                messages.append(f'setpoint.{key}: legacy metadata, not an effective new-run setting; specify initialization/storage explicitly')
    ga = data.get('ga', data.get('trajectory', {}).get('ga', {}))
    for name in preset['search']:
        if name in ga:
            preset['search'][name] = ga[name]
    preset['runtime']['workers'] = ga.get('workers', 1)
    if kind == 'D':
        preset['protocol']['trajectory'] = {k: deepcopy(v) for k, v in data['trajectory'].items()
                                            if k in NESTED_KEYS['protocol.trajectory']}
        preset['genome']['command']['coefficient_bound_nA'] = data['trajectory']['coefficient_bound_nA']
        messages.append('D sources must be supplied explicitly; historical default_sources is not applied')
    preset['preset'] = f'legacy:{source or "embedded controller"}'
    return preset, messages


def _integer(value, minimum, path):
    """Validate actual integers, excluding bool and integral-looking floats."""
    if type(value) is not int or value < minimum:
        raise ValueError(f'{path}: expected an integer >= {minimum}')


def _number(value, path):
    """Reject bool, strings and nonfinite numbers in physical/search settings."""
    if type(value) not in (float, int) or not np.isfinite(value):
        raise ValueError(f'{path}: expected a finite JSON number')


def controller_config(data):
    """Build the existing numerical domain from resolved user settings."""
    kind = data['experiment']
    result = dict(schema=GENOME_SCHEMA, experiment=EXPERIMENT_NAMES[kind], dt_ms=data['model']['dt_ms'],
                  mechanics=deepcopy(data['model']['mechanics']),
                  parameters=deepcopy(data['genome']['parameters']),
                  mirror_symmetry=data['genome']['mirror_symmetry'])
    if kind != 'D':
        result['simulation'] = deepcopy(data['protocol']['balance'])
        result['simulation']['replay_angles_deg'] = data['protocol']['replay']['angles_deg']
    if kind == 'C':
        result['setpoint'] = {**deepcopy(data['protocol']['setpoint']), **data['genome']['command'],
                              'checkpoints': data['search']['checkpoints'], 'save_figures': False}
    if kind == 'D':
        result['trajectory'] = {**deepcopy(data['protocol']['trajectory']),
                                'coefficient_bound_nA': data['genome']['command']['coefficient_bound_nA']}
    return GenomeConfig(result)


def resolve_config(path=None, *, overrides=None, action='train', solution=None):
    """Expand one preset and validate the final run without evaluating dynamics.

    JSON paths are relative to their declaring file; CLI paths are relative to
    cwd. A finetune/replay without a config inherits the solution's saved domain.
    """
    sources, paths, messages = {}, {}, []
    if path is None:
        if solution is None:
            raise ValueError('--config is required; no implicit training is started')
        from .artifacts import load_solution
        saved = load_solution(solution)
        if saved.effective_config:
            data = deepcopy(saved.effective_config)
            for key in ('controller', 'inspection', 'provenance', 'action'):
                data.pop(key, None)
            data['storage']['output'] = None
        else:
            data, messages = legacy_config(saved.config.data, str(solution))
            if saved.kind == 'polynomial':
                data['genome']['command']['degree'] = saved.command.shape[1]-1
            data['initialization']['current_tau_s'] = saved.current_tau_s
        sources = {key: f'solution:{Path(solution).resolve()}' for key, _ in _leaves(data)}
        data['initialization']['solution'] = str(Path(solution).resolve())
        data['initialization']['sources'] = []
        data['initialization']['extra_solutions'] = []
        data['initialization']['prepare'] = dict(clip_fields=[], values={}, mirror_projection=False, gain_match=False)
        data['initialization']['current_tau_s'] = saved.current_tau_s
        if action == 'finetune':
            defaults = dict(search=dict(method='joint', population=50, generations=200, seed=43,
                             mutation_probability=.3, sigma_start=.2, sigma_end=.015,
                             validation_every=20, checkpoints=[], table_generations=None, polarity_seeds=True),
                            runtime={'workers':8})
            _merge(data, defaults, 'operation:finetune defaults', sources)
    else:
        path = Path(path).expanduser().resolve()
        request = read_json(path)
        if request.get('schema') == GENOME_SCHEMA:
            data, messages = legacy_config(request, str(path))
            sources = {key: f'legacy:{path}' for key, _ in _leaves(data)}
            _resolve_paths(data, path.parent, paths, str(path))
        else:
            _validate_keys(request, full='controller' in request)
            if request.get('schema') != SCHEMA:
                raise ValueError(f'schema: expected {SCHEMA}; old controller configs use {GENOME_SCHEMA}')
            if 'controller' in request:
                data = {k: deepcopy(v) for k, v in request.items() if k not in ('controller', 'inspection', 'provenance', 'action')}
                sources = {key: f'effective_config:{path}' for key, _ in _leaves(data)}
                _resolve_paths(data, path.parent, paths, str(path))
            else:
                name = request.get('preset')
                if not isinstance(name, str) or Path(name).name != name:
                    raise ValueError('preset: specify a local versioned preset name')
                preset_path = project_root()/'configs/presets'/f'{name}.json'
                if not preset_path.is_file():
                    raise ValueError(f'preset: unknown {name!r}')
                data = read_json(preset_path)
                _validate_keys(data)
                _resolve_paths(data, preset_path.parent, paths, f'preset:{name}')
                sources = {key: f'preset:{name}' for key, _ in _leaves(data)}
                _resolve_paths(request, path.parent, paths, str(path))
                _merge(data, request, f'config:{path}', sources)
    override = deepcopy(overrides or {})
    if solution is not None:
        override.setdefault('initialization', {})['solution'] = str(Path(solution).expanduser().resolve())
    _validate_keys(override)
    _resolve_paths(override, Path.cwd(), paths, 'cli')
    _merge(data, override, 'cli', sources)
    data['action'] = action
    # The operation determines whether generation zero is a candidate.
    if action == 'finetune':
        data['protocol']['selection'] = 'initial_checkpoint_validation_v1'
        sources['protocol.selection'] = 'operation:finetune'
        if data['search']['method'] == 'compare':
            data['search']['method'] = 'joint'
            sources['search.method'] = 'operation:finetune'
        data['initialization']['strategy'] = 'standard'
    if solution is not None and action in ('replay', 'prepare'):
        data['initialization']['sources'] = []
    _validate_effective(data)
    data['provenance'] = dict(sources=sources, paths=paths, compatibility_messages=messages)
    if data['storage']['output'] is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        data['storage']['output'] = str(project_root()/'results'/f'{data["experiment"]}_{data["search"]["method"]}_{stamp}')
        sources['storage.output'] = 'generated:project/results'
    data['controller'] = controller_config(data).data
    return data


def _validate_effective(data):
    """Validate scientific applicability, types and scalar limits of a full request."""
    kind, search = data['experiment'], data['search']
    if kind not in EXPERIMENT_NAMES:
        raise ValueError('experiment: expected A, B, C or D')
    model = data['model']
    if model['name'] != MODEL or model['dt_ms'] != .1 or model['architecture'] != 'two_neurons_one_limb_v1':
        raise ValueError('model: this schema requires v3-tensile, 0.1 ms and two_neurons_one_limb_v1; '
                         'a larger network/MultyLimbSystem requires a new schema and model adapter')
    allowed = {'A': {'joint'}, 'B': {'joint'}, 'C': {'joint', 'staged'}, 'D': {'joint', 'command_only', 'compare'}}[kind]
    if search['method'] not in allowed:
        raise ValueError(f'search.method: {kind} supports {sorted(allowed)}')
    for key, minimum in [('population', 3), ('generations', 1), ('seed', 0), ('validation_every', 1)]:
        _integer(search[key], minimum, f'search.{key}')
    _integer(data['runtime']['workers'], 1, 'runtime.workers')
    _integer(data['storage']['recording_stride'], 1, 'storage.recording_stride')
    from .evolution import Schedule
    for key in ('sigma_start', 'sigma_end', 'mutation_probability'):
        _number(search[key], f'search.{key}')
    Schedule(search['sigma_start'], search['sigma_end'], search['mutation_probability'], search['mutation_probability'])
    for name in ('save_generations', 'replay', 'figures'):
        if type(data['storage'][name]) is not bool:
            raise ValueError(f'storage.{name}: expected a boolean')
    if type(data['genome']['mirror_symmetry']) is not bool:
        raise ValueError('genome.mirror_symmetry: expected a boolean')
    fields = data['genome']['parameters']
    if data['genome']['schema'] != GENOME_SCHEMA:
        raise ValueError(f'genome.schema: expected {GENOME_SCHEMA}')
    if len(fields) != len(NAMES) or tuple(f['name'] for f in fields) != NAMES:
        raise ValueError('genome.parameters: structural changes require a new genome schema')
    for field in fields:
        _check_keys(field, {'index', 'name', 'description', 'system_parameter', 'unit', 'units', *FIELD_KEYS},
                    f'genome.parameters.{field.get("name", "?")}')
        update = data['genome']['overrides'].get(field['name'], {})
        if update.get('mutable') is False and 'value' not in update:
            raise ValueError(f'genome.overrides.{field["name"]}: fixing a field requires value')
        if update.get('mutable') is True and not field['mutable']:
            if not {'minimum', 'maximum', 'scale'} <= set(update):
                raise ValueError(f'genome.overrides.{field["name"]}: freeing a field requires minimum/maximum/scale')
        field.update(update)
        for key in ('value', 'minimum', 'maximum'):
            if key in field:
                _number(field[key], f'genome.{field["name"]}.{key}')
        if type(field['mutable']) is not bool:
            raise ValueError(f'{field["name"]}.mutable: expected a boolean')
    prepared = data['initialization']['prepare']
    for key in ('mirror_projection', 'gain_match'):
        if type(prepared[key]) is not bool:
            raise ValueError(f'initialization.prepare.{key}: expected a boolean')
    if not isinstance(prepared['clip_fields'], list) or not isinstance(prepared['values'], dict):
        raise ValueError('prepare requires a clip_fields array and a values object')
    for name, value in prepared['values'].items():
        _number(value, f'initialization.prepare.values.{name}')
    if any(prepared.values()) and data['action'] not in ('prepare', 'validate'):
        raise ValueError('initialization.prepare applies only to prepare/validate')
    # An explicitly requested mirror preparation may project default center
    # values as well as the source. Bounds and mutable flags must already match.
    if prepared['mirror_projection'] and data['genome']['mirror_symmetry']:
        from .genome import MIRROR_INDICES
        values = [field['value'] for field in fields]
        for i, field in enumerate(fields):
            field['value'] = .5*(values[i]+values[MIRROR_INDICES[i]])
    command = data['genome']['command']
    if kind in 'AB' and command:
        raise ValueError('genome.command is inapplicable to zero-current balance A/B')
    if kind == 'C' and command['degree'] not in (2, 3):
        raise ValueError('genome.command.degree: C supports degrees 2 or 3')
    if kind == 'D' and command['degree'] != 3:
        raise ValueError('genome.command.degree: D requires the cubic representation')
    if kind in 'CD':
        if not np.isfinite(command['coefficient_bound_nA']) or command['coefficient_bound_nA'] <= 0:
            raise ValueError('genome.command.coefficient_bound_nA: expected positive finite bound')
    if kind != 'C' and data['protocol']['setpoint']:
        raise ValueError('protocol.setpoint applies only to C')
    if kind != 'D' and data['protocol']['trajectory']:
        raise ValueError('protocol.trajectory applies only to D')
    if kind == 'D' and data['protocol']['balance']:
        raise ValueError('protocol.balance is inapplicable to D')
    expected = 'trajectory_v1' if kind == 'D' else 'balance_v1'
    if data['protocol']['criterion'] != expected:
        raise ValueError(f'protocol.criterion: {kind} requires {expected}')
    if data['protocol']['selection'] not in POLICIES:
        raise ValueError('protocol.selection: unknown selection policy')
    expected_policy = ('d_comparison_validation_v1' if kind == 'D' and search['method'] == 'compare' else
                       'initial_checkpoint_validation_v1' if kind == 'D' else
                       'checkpoint_validation_v1' if kind == 'C' else 'final_validation_v1')
    if data['action'] in ('train', 'sweep') and data['protocol']['selection'] != expected_policy:
        raise ValueError(f'protocol.selection: {kind}/{search["method"]} requires {expected_policy}; '
                         'use finetune for generation-zero selection')
    if data['storage']['figures'] and not data['storage']['replay']:
        raise ValueError('storage.figures requires storage.replay; figures use saved traces')
    if kind in 'AB' and search['polarity_seeds'] and data['action'] not in ('finetune', 'validate'):
        raise ValueError('search.polarity_seeds is a finetune initialization; standard A/B starts unchanged')
    if kind != 'D' and data['initialization']['current_tau_s'] != 0:
        raise ValueError('initialization.current_tau_s applies only to D')
    if search['method'] != 'staged' and search['table_generations'] is not None:
        raise ValueError('search.table_generations applies only to C staged')
    if search['method'] == 'staged':
        if search['table_generations'] is None:
            search['table_generations'] = search['generations']//2
        _integer(search['table_generations'], 1, 'search.table_generations')
        if search['table_generations'] >= search['generations']:
            raise ValueError('C staged must allocate at least one generation to each stage')
    if not isinstance(search['checkpoints'], list) or any(type(g) is not int or g < 1 for g in search['checkpoints']):
        raise ValueError('search.checkpoints: expected positive integer generations')
    initial = data['initialization']
    if initial['solution'] and initial['sources']:
        raise ValueError('initialization: choose one solution or a source comparison, not both')
    if initial['sources'] and (kind != 'D' or search['method'] != 'compare'):
        raise ValueError('initialization.sources applies only to standard D comparison')
    if initial['strategy'] not in ('standard', 'local_d'):
        raise ValueError('initialization.strategy: expected standard or local_d')
    if not np.isfinite(initial['current_tau_s']) or initial['current_tau_s'] < 0:
        raise ValueError('initialization.current_tau_s: expected finite time >= 0')
    if data['action'] in ('train', 'finetune', 'sweep', 'validate') and kind == 'D':
        if search['method'] == 'compare' and not initial['sources']:
            raise ValueError('initialization.sources: D comparison requires explicit compatible C/D solutions')
        if search['method'] != 'compare' and not initial['solution']:
            raise ValueError('initialization.solution: D requires an explicit compatible C/D solution')
    if initial['strategy'] == 'local_d' and kind != 'D':
        raise ValueError('initialization.strategy=local_d applies only to D')
    _integer(initial['candidate_seed_offset'], 0, 'initialization.candidate_seed_offset')
    if not np.isfinite(initial['local_sigma']) or initial['local_sigma'] <= 0:
        raise ValueError('initialization.local_sigma: expected a positive finite value')
    for key in ('evaluate_test',):
        if type(data['protocol'][key]) is not bool:
            raise ValueError(f'protocol.{key}: expected a boolean')
    if type(search['polarity_seeds']) is not bool:
        raise ValueError('search.polarity_seeds: expected a boolean')
    for bank in ('balance', 'trajectory'):
        settings = data['protocol'][bank]
        if not settings:
            continue
        for name in ('training', 'validation', 'test'):
            _integer(settings[name+'_count'], 4 if bank == 'balance' else 2,
                     f'protocol.{bank}.{name}_count')
            _integer(settings['bank_seeds'][name], 0, f'protocol.{bank}.bank_seeds.{name}')
    if kind == 'D':
        settings = data['protocol']['trajectory']
        for name in ('hold_s', 'ramp_s'):
            if not np.isfinite(settings[name]) or settings[name] < 0:
                raise ValueError(f'protocol.trajectory.{name}: expected a nonnegative finite time')
        if not settings['filter_times_s'] or any(not np.isfinite(t) or t < 0 for t in settings['filter_times_s']):
            raise ValueError('protocol.trajectory.filter_times_s: expected nonnegative finite times')
    replay = data['protocol']['replay']
    for name in ('angles_deg', 'speeds_rad_s', 'targets_deg'):
        values = replay[name]
        if kind == 'D' and values is None:
            continue
        if kind == 'D' and name == 'targets_deg':
            raise ValueError('protocol.replay.targets_deg is inapplicable to D; references are trajectory specifications')
        if not isinstance(values, list) or not values or not np.isfinite(values).all():
            raise ValueError(f'protocol.replay.{name}: expected a nonempty finite list')
        if name in ('angles_deg', 'targets_deg') and any(abs(v) >= 90 for v in values):
            raise ValueError(f'protocol.replay.{name}: angles must be strictly inside ±90 degrees')
        if kind == 'C' and name == 'targets_deg' and any(abs(v) > 45 for v in values):
            raise ValueError('protocol.replay.targets_deg: C targets must be within ±45 degrees')
    if kind in 'AB' and replay['targets_deg'] != [0.]:
        raise ValueError('protocol.replay.targets_deg: A/B hold the zero target')
    if replay['duration_s'] is not None:
        from ..D_trajectory.trajectory import step_count
        step_count(replay['duration_s'])
    for name in ('solution',):
        if initial[name] and not Path(initial[name]).is_file():
            raise ValueError(f'initialization.{name}: file not found: {initial[name]}')
    for name in ('sources', 'extra_solutions'):
        for path in initial[name]:
            if not Path(path).is_file():
                raise ValueError(f'initialization.{name}: file not found: {path}')
    axes = data['sweep']['axes']
    for key, values in axes.items():
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f'sweep.axes.{key}: expected nonempty unique values')
    if data['action'] == 'sweep' and not axes:
        raise ValueError('sweep.axes: at least one explicit axis is required')
    # Validate domain values and geometry with the exact historical codec.
    controller_config(data)
