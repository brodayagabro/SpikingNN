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
    """Return a SHA256 without loading a large result into memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
        unit = self.encode(physical)
        write_json(path, dict(schema=SCHEMA, configuration=self.data,
                             configuration_sha256=self.signature, names=list(NAMES),
                             physical=np.asarray(physical).tolist(), normalized=unit.tolist(),
                             metadata=metadata))


def load_genome(path):
    """Load a current genome and return its configuration and physical values.

    Historical arrays are rejected: their offsets 46/47 mean applied currents.
    Use ``import_legacy`` explicitly for a compatible historical controller.
    """
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != SCHEMA or tuple(payload.get("names", ())) != NAMES:
        raise ValueError("Not a current genome; historical files need explicit import_legacy")
    config = GenomeConfig(payload["configuration"])
    if payload["configuration_sha256"] != config.signature:
        raise ValueError("Saved genome configuration checksum mismatch")
    physical = np.asarray(payload["physical"], dtype=float)
    np.testing.assert_allclose(config.encode(physical), payload["normalized"], atol=1e-12, rtol=0)
    return config, physical


def import_legacy(path, config):
    """Explicitly convert an old zero-current JSON genome to the new schema.

    Preserve all 50 controller values, replace old zero-current slots with the
    configured geometry, and reject out-of-domain values instead of clipping.
    This does not transfer historical performance claims to a new protocol.
    """
    payload = json.loads(Path(path).read_text())
    genes = payload.get("genes", [])
    expected_names = list(NAMES)
    expected_names[46:48] = ["input.bias_nA.0", "input.bias_nA.1"]
    if (payload.get("version") != 1 or len(genes) != SIZE
            or [gene.get("name") for gene in genes] != expected_names):
        raise ValueError("Expected a historical version-1 genome with applied-current slots")
    physical = np.array([g["value"] for g in genes], dtype=float)
    if np.any(physical[46:48] != 0):
        raise ValueError("Historical applied currents must be supplied separately")
    physical[46:48] = config.initial_physical[46:48]
    config.encode(physical)
    return physical
