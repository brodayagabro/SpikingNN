"""The explicit 52-field genome: 50 controller parameters and two fixed lengths.

Applied currents are simulation inputs, not genes. Every saved genome embeds its
configuration, so geometry and search bounds survive outside the experiment folder.
Controller fields may also be fixed: B keeps both muscle A values at 0.37,
leaving 48 mutable coordinates while retaining all 52 physical fields.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "afferent_controller_52_v4"
SIZE = 52
WEIGHT_INDICES = np.array([16, 23])
GEOMETRY_INDICES = np.array([46, 47])

# 0:8    Izhikevich a, b, c, d, each for neurons 0 and 1.
# 8:12   Recurrent W[target, source]; 12:16 decay times in milliseconds.
# 16:23  Flexor w, N, A, tau_c, tau_1, m, k; 23:30 the same for extensor.
# 30:42  Q_aff[neuron, channel], channels Ia_F, II_F, Ib_F, Ia_E, II_E, Ib_E.
# 42:44  Receptor length thresholds (m); 44:46 force thresholds (N).
# 46:48  a1 and a2 (m): fixed config values, excluded from mutation/crossover.
# 48:52  P[muscle, neuron]. Mass/length/viscosity/gravity remain config metadata.
NAMES = (
    *(f"neurons.{key}.{i}" for key in ("a", "b", "c", "d") for i in range(2)),
    *(f"synapses.W.{i}.{j}" for i in range(2) for j in range(2)),
    *(f"synapses.tau_ms.{i}.{j}" for i in range(2) for j in range(2)),
    *(f"muscles.{side}.{key}" for side in ("flexor", "extensor")
      for key in ("w", "N", "A", "tau_c", "tau_1", "m", "k")),
    *(f"feedback.Q_aff.{i}.{channel}" for i in range(2)
      for channel in ("Ia_F", "II_F", "Ib_F", "Ia_E", "II_E", "Ib_E")),
    *(f"feedback.L_th.{i}" for i in range(2)),
    *(f"feedback.F_th.{i}" for i in range(2)),
    "geometry.a1", "geometry.a2",
    *(f"output.P.{i}.{j}" for i in range(2) for j in range(2)),
)
MECHANICAL_NAMES = ("mass_kg", "length_m", "viscosity", "gravity_m_s2")
GENE_INDICES = {name: index for index, name in enumerate(NAMES)}
MUSCLE_INDICES = np.array([[GENE_INDICES[f'muscles.{side}.{key}']
                            for key in ('w', 'N', 'A', 'tau_c', 'tau_1', 'm', 'k')]
                           for side in ('flexor', 'extensor')])
OUTPUT_INDICES = np.array([[GENE_INDICES[f'output.P.{i}.{j}'] for j in range(2)] for i in range(2)])
FEEDBACK_INDICES = np.array([[GENE_INDICES[f'feedback.Q_aff.{i}.{channel}']
                              for channel in ('Ia_F', 'II_F', 'Ib_F', 'Ia_E', 'II_E', 'Ib_E')]
                             for i in range(2)])

# Reflection swaps neurons and muscles simultaneously. Matrix connections swap
# BOTH axes; afferent channels swap the two three-channel muscle blocks.
# Geometry a1/a2 are distinct attachment distances, not a flexor/extensor pair.
MIRROR_INDICES = np.array([
    1, 0, 3, 2, 5, 4, 7, 6, 11, 10, 9, 8, 15, 14, 13, 12,
    *range(23, 30), *range(16, 23),
    39, 40, 41, 36, 37, 38, 33, 34, 35, 30, 31, 32,
    43, 42, 45, 44, 46, 47, 51, 50, 49, 48,
])


def project_mirror(physical):
    """Average each reflected physical pair; leave both geometry values intact.

    This is an explicit warm-start projection, not an operation performed while
    loading historical genomes. It does not preserve the original performance.
    """
    p = GenomeConfig.validate_physical(physical)
    return .5 * (p + p[MIRROR_INDICES])


def write_json(path, value):
    """Write finite UTF-8 JSON atomically, preserving valid numerical metadata."""
    path = Path(path)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    """Return a streaming SHA-256, including on Python 3.10."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class GenomeConfig:
    """Load named parameters and map mutable coordinates to physical values.

    ``decode`` inserts all fixed fields after genetic operations. ``encode`` rejects
    fixed-value mismatches and values outside this experiment's physical bounds.
    Optional ``mirror_symmetry`` ties reflected fields to one search coordinate;
    ``free`` lists independent representatives and ``mutable`` lists both members.
    All configuration values are copied; callers cannot change a running domain
    by editing the input dictionary.
    """

    def __init__(self, source):
        """Read a JSON path or configuration dictionary and validate the schema."""
        self.data = deepcopy(source if isinstance(source, dict)
                             else json.loads(Path(source).read_text()))
        data = self.data
        if data.get("schema") != SCHEMA or data.get("dt_ms") != .1:
            raise ValueError("Expected the new 52-field schema and dt_ms=0.1")
        fields = data["parameters"]
        if len(fields) != SIZE or tuple(p["name"] for p in fields) != NAMES:
            raise ValueError("Genome field names/order must match the explicit 52-field schema")
        if [p["index"] for p in fields] != list(range(SIZE)):
            raise ValueError("Genome indices must be 0..51")
        self.initial_physical = np.array([p["value"] for p in fields], dtype=float)
        if not np.isfinite(self.initial_physical).all():
            raise ValueError("Nonfinite genome value")
        if any(fields[i]["mutable"] for i in GEOMETRY_INDICES):
            raise ValueError("a1 and a2 are fixed hyperparameters and cannot mutate")
        self.mirror_symmetry = data.get("mirror_symmetry", False)
        if not isinstance(self.mirror_symmetry, bool):
            raise ValueError("mirror_symmetry must be a boolean")
        self.mutable = np.array([i for i, p in enumerate(fields) if p["mutable"]], dtype=int)
        if self.mirror_symmetry:
            for i, j in enumerate(MIRROR_INDICES):
                for key in ("mutable", "minimum", "maximum", "scale"):
                    if fields[i].get(key) != fields[j].get(key):
                        raise ValueError(f"Reflected fields require identical {key}: {NAMES[i]}, {NAMES[j]}")
            self.free = np.array([i for i in self.mutable if i <= MIRROR_INDICES[i]], dtype=int)
        else:
            self.free = self.mutable.copy()
        if not len(self.free):
            raise ValueError("At least one controller parameter must be mutable")
        self.log = np.array([fields[i].get("scale") == "log" for i in self.free])
        self.minimum = np.array([fields[i]["minimum"] for i in self.free], dtype=float)
        self.maximum = np.array([fields[i]["maximum"] for i in self.free], dtype=float)
        if (not np.isfinite(self.minimum).all() or not np.isfinite(self.maximum).all()
                or np.any(self.minimum >= self.maximum)
                or np.any(self.minimum[self.log] <= 0)
                or any(fields[i].get("scale") not in ("linear", "log") for i in self.free)):
            raise ValueError("Invalid finite search bounds or logarithmic scale")
        self.lower, self.upper = self.minimum.copy(), self.maximum.copy()
        self.lower[self.log] = np.log(self.lower[self.log])
        self.upper[self.log] = np.log(self.upper[self.log])
        self.mechanics = np.array([data["mechanics"][key] for key in MECHANICAL_NAMES], dtype=float)
        if (not np.isfinite(self.mechanics).all() or np.any(self.mechanics[[0, 1, 3]] <= 0)
                or self.mechanics[2] < 0):
            raise ValueError("Invalid mass, link length, viscosity or gravity")
        self.validate_physical(self.initial_physical)
        self.encode(self.initial_physical)
        expected_cap = {"A_balance": 1., "B_balance": .1}.get(data.get("experiment"))
        if expected_cap is not None:
            for i in WEIGHT_INDICES:
                field = fields[i]
                cap = field["maximum"] if field["mutable"] else field["value"]
                # Older B snapshots with a 0.02 cap remain readable. The live
                # JSON defines the restored lower bound and strict upper bound.
                strict = data['experiment'] == 'B_balance'
                if cap > expected_cap or (strict and cap == expected_cap):
                    relation = '<' if strict else '<='
                    raise ValueError(f"{data['experiment']} requires w {relation} {expected_cap}")

    @property
    def size(self):
        """Return independent coordinates, excluding fixed fields and tied duplicates."""
        return len(self.free)

    @property
    def signature(self):
        """Return a content signature for bounds, mechanics and the entire protocol."""
        payload = json.dumps(self.data, sort_keys=True, allow_nan=False).encode()
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def validate_physical(physical):
        """Reject physically invalid parameters before reaching compiled dynamics."""
        p = np.asarray(physical, dtype=float)
        if p.shape != (SIZE,) or not np.isfinite(p).all():
            raise ValueError("Expected 52 finite physical parameters")
        positive = [0, 1, *range(12, 16), *range(16, 30), *range(42, 48)]
        if np.any(p[positive] <= 0) or np.any(p[48:52] < 0):
            raise ValueError("Times, muscle parameters and geometry must be positive; P >= 0")
        if min(p[19], p[20], p[26], p[27]) < .1 or p[46] == p[47]:
            raise ValueError("Muscle times must be >= 0.1 ms and a1 must differ from a2")
        return p

    def initial(self):
        """Return the normalized initial candidate from this experiment's JSON."""
        return self.encode(self.initial_physical)

    def decode(self, unit):
        """Map free [0,1] coordinates to 52 physical values, inserting fixed fields."""
        unit = np.asarray(unit, dtype=float)
        if unit.shape != (self.size,) or not np.isfinite(unit).all() or np.any((unit < 0) | (unit > 1)):
            raise ValueError(f"Expected {self.size} normalized coordinates in [0,1]")
        values = self.lower + unit * (self.upper - self.lower)
        values[self.log] = np.exp(values[self.log])
        p = self.initial_physical.copy()
        # Clip exp/log roundoff so inclusive upper bounds are respected exactly.
        p[self.free] = np.clip(values, self.minimum, self.maximum)
        if self.mirror_symmetry:
            p[MIRROR_INDICES[self.free]] = p[self.free]
        return p

    def encode(self, physical):
        """Encode values after checking search bounds and exact fixed hyperparameters."""
        p = self.validate_physical(physical)
        fixed = np.setdiff1d(np.arange(SIZE), self.mutable)
        if not np.array_equal(p[fixed], self.initial_physical[fixed]):
            raise ValueError("Fixed parameters differ from genome_cfg.json")
        if self.mirror_symmetry and not np.array_equal(p, p[MIRROR_INDICES]):
            raise ValueError("Physical parameters violate exact mirror symmetry")
        values = p[self.free].copy()
        if np.any(values < self.minimum) or np.any(values > self.maximum):
            raise ValueError("Physical values exceed this experiment's bounds")
        values[self.log] = np.log(values[self.log])
        return np.clip((values - self.lower) / (self.upper - self.lower), 0, 1)

    def save(self, path, physical, **metadata):
        """Save a self-contained genome with descriptions, geometry and provenance."""
        write_json(path, self.payload(physical, **metadata))

    def payload(self, physical, **metadata):
        """Return the portable legacy document without performing file I/O."""
        unit = self.encode(physical)
        return dict(schema=SCHEMA, configuration=self.data,
                             configuration_sha256=self.signature, names=list(NAMES),
                             physical=np.asarray(physical).tolist(), normalized=unit.tolist(),
                             metadata=metadata)


def load_genome(path):
    """Load a current genome and return its configuration and physical values.

    Historical arrays are rejected: their offsets 46/47 mean applied currents.
    Use ``import_legacy`` explicitly for a compatible historical controller.
    """
    payload = path if isinstance(path, dict) else json.loads(Path(path).read_text())
    if payload.get("schema") != SCHEMA or tuple(payload.get("names", ())) != NAMES:
        raise ValueError("Expected a 52-field genome with embedded configuration")
    config = GenomeConfig(payload["configuration"])
    if payload["configuration_sha256"] != config.signature:
        raise ValueError("Saved genome configuration checksum mismatch")
    physical = np.asarray(payload["physical"], dtype=float)
    np.testing.assert_allclose(config.encode(physical), payload["normalized"], atol=1e-12, rtol=0)
    return config, physical


class BalanceGenome:
    """Expose the same population interface as JointGenome, with zero commands."""

    kind = 'none'

    def __init__(self, config):
        self.controller = config if isinstance(config, GenomeConfig) else GenomeConfig(config)
        if self.controller.data['experiment'] not in ('A_balance', 'B_balance'):
            raise ValueError('BalanceGenome requires an A or B configuration')
        self.size, self.names = self.controller.size, NAMES

    def decode(self, unit):
        return self.controller.decode(unit), np.zeros((2, 0))

    def encode(self, physical, command=None):
        return self.controller.encode(physical)

    def currents(self, command, cases):
        return np.zeros((len(cases), 2))

    def payload(self, unit, **metadata):
        payload = self.controller.payload(self.controller.decode(unit), **metadata)
        # Preserve the evaluated coordinates bit-for-bit: exp/log re-encoding
        # can introduce roundoff and should not alter a frozen solution.
        payload['normalized'] = np.asarray(unit, dtype=float).tolist()
        return payload

    def save(self, path, unit, **metadata):
        write_json(path, self.payload(unit, **metadata))



JOINT_SCHEMA = 'afferent_setpoint_joint_v1'

def polynomial_currents(coefficients, target_rad):
    """Evaluate two polynomials in target/45 degrees, including constant terms.

    Coefficients have shape (2, degree+1), with degree 0, 2 or 3. The resulting
    two currents remain constant throughout an episode; no state feedback is added.
    """
    coefficients = np.asarray(coefficients, dtype=float)
    if (coefficients.ndim != 2 or coefficients.shape[0] != 2
            or coefficients.shape[1] not in (1, 3, 4)
            or not np.isfinite(coefficients).all()
            or not np.isfinite(target_rad) or abs(target_rad) >= np.pi/2):
        raise ValueError("Expected two finite constant/quadratic/cubic coefficient rows and a valid target")
    x = target_rad / np.deg2rad(45.)
    return coefficients @ (x ** np.arange(coefficients.shape[1]))



class JointGenome:
    """Map one normalized chromosome to a controller and two command functions.

    Parameters
    ----------
    config : GenomeConfig, path or dict
        C controller domain and independent multi-target protocol.
    kind : {'polynomial', 'table'}
        Append ascending-power coefficients or one current per training target.
    degree : {2, 3}
        Polynomial degree; table width is the number of training targets.
    """

    def __init__(self, config, kind='polynomial', degree=3):
        """Validate the controller and append an explicitly named command block."""
        self.controller = config if isinstance(config, GenomeConfig) else GenomeConfig(config)
        if self.controller.data['experiment'] != 'C_setpoint_balance':
            raise ValueError('Joint training belongs to C_setpoint_balance')
        if kind not in ('polynomial', 'table') or degree not in (2, 3):
            raise ValueError('Use a table or a degree-2/3 polynomial')
        self.kind, self.degree = kind, degree
        settings = self.controller.data['setpoint']
        self.targets = np.asarray(settings['targets_deg'], dtype=float)
        if (self.targets.ndim != 1 or len(self.targets) < 5
                or not np.isfinite(self.targets).all() or np.any(np.diff(self.targets) <= 0)
                or self.targets[0] != -45 or self.targets[-1] != 45):
            raise ValueError('Use at least five increasing training targets spanning -45 to +45 degrees')
        self.width = degree+1 if kind == 'polynomial' else len(self.targets)
        if self.controller.mirror_symmetry and not np.array_equal(self.targets, -self.targets[::-1]):
            raise ValueError('Mirror symmetry requires a reflected target table')
        self.bound = float(settings['coefficient_bound_nA' if kind == 'polynomial' else 'table_current_bound_nA'])
        if not np.isfinite(self.bound) or self.bound <= 0:
            raise ValueError('Command bounds must be finite and positive')
        suffixes = range(self.width) if kind == 'polynomial' else self.targets
        self.names = (*NAMES, *(f'command.{kind}.neuron_{i}.{k:g}' for i in range(2) for k in suffixes))

    @property
    def size(self):
        """Return independent controller and command coordinates, honoring reflection."""
        return self.controller.size + (1 if self.controller.mirror_symmetry else 2)*self.width

    @property
    def specification(self):
        """Return the portable definition of the appended command genes."""
        spec = dict(kind=self.kind, degree=self.degree, targets_deg=self.targets.tolist(),
                    bound_nA=self.bound, variable='target_rad / radians(45)', order='neuron-major')
        if self.controller.mirror_symmetry:
            spec['mirror_symmetry'] = True
        return spec

    def reflected_command(self, first):
        """Return neuron 1's command from I1(theta) = I0(-theta)."""
        return first * (-1.)**np.arange(self.width) if self.kind == 'polynomial' else first[::-1]

    def encode(self, physical, command):
        """Validate and concatenate controller and command coordinates in [0,1]."""
        command = np.asarray(command, dtype=float)
        if command.shape != (2,self.width) or not np.isfinite(command).all() or np.any(abs(command) > self.bound):
            raise ValueError('Command shape or coefficient/current bounds are invalid')
        if self.controller.mirror_symmetry:
            if not np.allclose(command[1], self.reflected_command(command[0]), rtol=0, atol=1e-10):
                raise ValueError('Commands violate I1(theta) = I0(-theta)')
            command = command[0]
        return np.r_[self.controller.encode(physical), .5+.5*command.ravel()/self.bound]

    def decode(self, unit):
        """Decode a chromosome, restoring every fixed controller hyperparameter."""
        unit = np.asarray(unit, dtype=float)
        if unit.shape != (self.size,) or not np.isfinite(unit).all() or np.any((unit<0)|(unit>1)):
            raise ValueError(f'Expected {self.size} finite normalized joint coordinates')
        command = (unit[self.controller.size:]*2-1)*self.bound
        command = (np.array([command, self.reflected_command(command)])
                   if self.controller.mirror_symmetry else command.reshape(2, self.width))
        return self.controller.decode(unit[:self.controller.size]), command

    def currents(self, command, cases):
        """Return K-by-2 constant commands; a table never interpolates unseen targets."""
        targets = np.asarray(cases)[:,0]
        if self.kind == 'polynomial':
            return np.array([polynomial_currents(command, target) for target in targets])
        match = np.isclose(targets[:,None], np.deg2rad(self.targets)[None,:], rtol=0, atol=1e-12)
        if not np.all(match.sum(axis=1)==1):
            raise ValueError('A current table can only replay its training target angles')
        return np.asarray(command)[:,np.argmax(match,axis=1)].T

    def save(self, path, unit, **metadata):
        """Save all physical genes, English mappings and the original controller domain."""
        write_json(path, self.payload(unit, **metadata))

    def payload(self, unit, **metadata):
        """Return the complete command/controller document without file I/O."""
        physical, command = self.decode(unit)
        fields = deepcopy(self.controller.data['parameters'])
        for field, value in zip(fields, physical):
            field['value'] = float(value)
        for i, name in enumerate(self.names[52:]):
            neuron, index = divmod(i,self.width)
            meaning = (f'Coefficient of (target_rad / radians(45))**{index}' if self.kind=='polynomial'
                       else f'Constant command at target {self.targets[index]:g} degrees')
            fields.append(dict(index=52+i, name=name, value=float(command.ravel()[i]), mutable=True,
                               minimum=-self.bound, maximum=self.bound, scale='linear', unit='model units',
                               description=f'{meaning} applied to neuron {neuron}; independent of measured state.'))
        return dict(schema=JOINT_SCHEMA, configuration=self.controller.data,
                   configuration_sha256=self.controller.signature, command=self.specification,
                   names=list(self.names), parameters=fields, physical=np.r_[physical,command.ravel()].tolist(),
                   normalized=np.asarray(unit).tolist(), metadata=metadata)


def load_solution(path):
    """Load a portable A/B/C genome and verify its configuration and coordinates.

    The embedded configuration governs replay. A nearby genome_cfg.json cannot
    silently change a published solution. Both original and new exports work.
    """
    data = deepcopy(path) if isinstance(path, dict) else json.loads(Path(path).read_text())
    if data.get('schema') == SCHEMA:
        config, physical = load_genome(data)
        return BalanceGenome(config), np.asarray(data['normalized'], dtype=float)
    if data.get('schema') != JOINT_SCHEMA:
        raise ValueError('Expected a portable balance or setpoint genome, not a raw parameter array')
    spec = data['command']
    genome = JointGenome(data['configuration'], spec['kind'], spec['degree'])
    if (genome.specification != spec or list(genome.names) != data['names']
            or genome.controller.signature != data['configuration_sha256']):
        raise ValueError('Command definition, parameter names or configuration checksum differs')
    unit = np.asarray(data['normalized'], dtype=float)
    physical, command = genome.decode(unit)
    np.testing.assert_allclose(np.r_[physical, command.ravel()], data['physical'], rtol=0, atol=1e-11)
    return genome, unit

