"""A stateful angle-command interface and portable trajectory genomes."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from ..core.genome import GenomeConfig, NAMES, write_json
from ..simulation import initial_state, write_state
from .trajectory import DT_S, command_currents, step_count, stream

SCHEMA = 'afferent_trajectory_v1'


class TrajectoryGenome:
    """Append eight cubic current-law coefficients to the 52 physical fields.

    Fixed geometry and muscle A values remain embedded but are never optimized.
    Lower degree C polynomials are represented exactly by zero cubic terms.
    """

    def __init__(self, config):
        """Read a D controller domain and its symmetric coefficient bound."""
        self.controller = config if isinstance(config, GenomeConfig) else GenomeConfig(config)
        if self.controller.data['experiment'] != 'D_trajectory':
            raise ValueError('Expected a D_trajectory configuration')
        self.bound = float(self.controller.data['trajectory']['coefficient_bound_nA'])
        if not np.isfinite(self.bound) or self.bound <= 0:
            raise ValueError('Expected a positive finite coefficient bound')
        self.names = (*NAMES, *(f'command.polynomial.neuron_{i}.{k}' for i in range(2) for k in range(4)))

    @property
    def size(self):
        """Return independent controller dimensions plus eight or four command genes."""
        return self.controller.size+(4 if self.controller.mirror_symmetry else 8)

    def encode(self, physical, coefficients):
        """Map physical values and ascending-power coefficients into [0, 1]."""
        c = np.asarray(coefficients, dtype=float)
        if c.shape != (2, 4) or not np.isfinite(c).all() or np.any(abs(c) > self.bound):
            raise ValueError('Expected 2-by-4 finite bounded polynomial coefficients')
        if self.controller.mirror_symmetry:
            if not np.allclose(c[1], c[0]*[1, -1, 1, -1], rtol=0, atol=1e-10):
                raise ValueError('Coefficients violate I1(theta) = I0(-theta)')
            c = c[0]
        return np.r_[self.controller.encode(physical), .5+.5*c.ravel()/self.bound]

    def decode(self, unit):
        """Restore fixed fields and decode the physical controller/current law."""
        unit = np.asarray(unit, dtype=float)
        if unit.shape != (self.size,) or not np.isfinite(unit).all() or np.any((unit < 0) | (unit > 1)):
            raise ValueError(f'Expected {self.size} finite coordinates in [0, 1]')
        c = (unit[self.controller.size:]*2-1)*self.bound
        c = (np.array([c, c*[1, -1, 1, -1]]) if self.controller.mirror_symmetry else c.reshape(2, 4))
        return self.controller.decode(unit[:self.controller.size]), c

    def save(self, path, unit, *, current_tau_s=0., **metadata):
        """Save a self-contained genome, English parameter mappings and provenance."""
        if not np.isfinite(current_tau_s) or current_tau_s < 0:
            raise ValueError('Invalid command filter time')
        p, c = self.decode(unit)
        fields = deepcopy(self.controller.data['parameters'])
        for field, value in zip(fields, p):
            field['value'] = float(value)
        for i, value in enumerate(c.ravel()):
            neuron, power = divmod(i, 4)
            fields.append(dict(index=52+i, name=self.names[52+i], value=float(value),
                               mutable=True, minimum=-self.bound, maximum=self.bound,
                               scale='linear', unit='nA',
                               description=f'Coefficient of (desired angle / 45 degrees)**{power}; no measured-state dependence.',
                               system_parameter=f'Applied current to neuron {neuron}'))
        write_json(path, dict(schema=SCHEMA, configuration=self.controller.data,
                              configuration_sha256=self.controller.signature,
                              names=list(self.names), parameters=fields,
                              physical=np.r_[p, c.ravel()].tolist(), normalized=np.asarray(unit).tolist(),
                              current_tau_s=current_tau_s, metadata=metadata))


def load_solution(path):
    """Read a saved C or D polynomial solution, returning physical inputs and metadata."""
    path = Path(path)
    data = json.loads(path.read_text())
    if data.get('schema') == SCHEMA:
        genome = TrajectoryGenome(data['configuration'])
        if genome.controller.signature != data['configuration_sha256'] or list(genome.names) != data['names']:
            raise ValueError('D genome checksum or names mismatch')
        p, c = genome.decode(data['normalized'])
        np.testing.assert_allclose(np.r_[p, c.ravel()], data['physical'], rtol=0, atol=1e-11)
        tau = float(data['current_tau_s'])
        cfg = genome.controller
    else:
        # The legacy C codec is needed only at the import boundary, never per step.
        from ..C_setpoint_balance.joint_study import load_joint_genome
        genome, unit = load_joint_genome(path)
        if genome.kind != 'polynomial':
            raise ValueError('Continuous angle commands require a polynomial, not a lookup table')
        p, old = genome.decode(unit)
        c = np.zeros((2, 4))
        c[:, :old.shape[1]] = old
        cfg, tau = genome.controller, 0.
    if not np.isfinite(tau) or tau < 0:
        raise ValueError('Invalid command filter time')
    return cfg, p, c, tau, data


class AngleController:
    """Continuously simulate the limb driven by a desired angle in degrees.

    ``step`` advances one 0.1 ms step; ``advance`` holds a command for a block.
    Both preserve neurons, synapses, calcium, muscle force, afferents and limb
    state. Only ``reset`` discards the history. The target is a command, not a
    guarantee that the actual limb angle equals it. This is a simulation input
    interface, not a hard real-time hardware scheduler.

    Parameters
    ----------
    genome_path : path-like
        A saved C polynomial genome or a D trajectory genome.
    initial_angle_deg, initial_velocity_rad_s : float
        Initial limb angle relative to the unstable vertical and angular speed.
    target_angle_deg : float
        Desired angle within the calibrated interval [-45, 45] degrees.
    current_tau_s : float, optional
        First-order current smoothing time; zero applies commands immediately.
        By default use the saved D setting (zero for C).
    """

    def __init__(self, genome_path, *, initial_angle_deg=0., initial_velocity_rad_s=0.,
                 target_angle_deg=0., current_tau_s=None):
        """Load a portable solution and explicitly initialize all dynamic state."""
        cfg, self._physical, self._coefficients, saved_tau, _ = load_solution(genome_path)
        self._mechanics = cfg.mechanics.copy()
        self._tau = saved_tau if current_tau_s is None else float(current_tau_s)
        if not np.isfinite(self._tau) or self._tau < 0:
            raise ValueError('current_tau_s must be finite and nonnegative')
        self.reset(initial_angle_deg, initial_velocity_rad_s, target_angle_deg)

    @property
    def target_angle_deg(self):
        """Return the desired angle; assigning it never resets physical state."""
        return self._target

    @target_angle_deg.setter
    def target_angle_deg(self, value):
        """Accept a finite desired angle inside the calibrated command domain."""
        value = float(value)
        if not np.isfinite(value) or abs(value) > 45:
            raise ValueError('target_angle_deg must be inside [-45, 45] degrees')
        self._target = value

    def reset(self, initial_angle_deg=0., initial_velocity_rad_s=0., target_angle_deg=0.):
        """Reset once; initialize filtered currents at the initial target command."""
        if (not np.isfinite([initial_angle_deg, initial_velocity_rad_s]).all()
                or abs(initial_angle_deg) >= 90):
            raise ValueError('Initial state must be finite and strictly inside +/-90 degrees')
        self.target_angle_deg = target_angle_deg
        self._state = initial_state(self._physical, np.deg2rad(initial_angle_deg), initial_velocity_rad_s)
        self._current = command_currents(self._coefficients, np.deg2rad(self._target))
        self._steps, self._failure = 0, 0
        return self.snapshot()

    def snapshot(self):
        """Return copied observations; editing them cannot mutate the simulator."""
        row = np.zeros(26)
        write_state(row, self._state, self._steps*DT_S)
        return dict(time_s=float(row[0]), target_angle_deg=self._target,
                    angle_deg=float(np.rad2deg(row[1])), velocity_rad_s=float(row[2]),
                    input_currents=self._current.copy(), membrane_mV=row[5:7].copy(),
                    recovery=row[7:9].copy(), force_N=row[3:5].copy(), calcium=row[9:11].copy(),
                    afferents=row[11:17].copy(), activation=row[17:19].copy(),
                    synapses=row[21:25].reshape(2, 2).copy(),
                    failure=('none', 'fall', 'nonfinite')[self._failure])

    def step(self, target_angle_deg=None):
        """Apply an optional new angle command and advance exactly 0.1 ms."""
        return self.advance(target_angle_deg, DT_S)

    def advance(self, target_angle_deg=None, duration_s=.01):
        """Hold a desired angle for duration_s seconds without resetting state.

        A fall or numerical failure stops integration and is reported in the
        snapshot. Further calls require an explicit reset. Timing refers to
        simulated time; the caller controls wall-clock command delivery.
        """
        steps = step_count(duration_s)
        if self._failure:
            raise RuntimeError('Simulation stopped; call reset before advancing again')
        if target_angle_deg is not None:
            self.target_angle_deg = target_angle_deg
        reference = np.zeros((steps+1, 2))
        reference[:, 0] = np.deg2rad(self._target)
        result = stream(self._physical, self._coefficients, self._mechanics,
                        self._state, self._current, reference, self._tau, 0, 0, self._steps)
        self._steps += result[4]
        self._failure = result[5]
        return self.snapshot()
