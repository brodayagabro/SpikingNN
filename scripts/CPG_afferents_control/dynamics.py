"""Numba transcription of the current SpikingNN API step; fixed dt=0.1 ms.

No target trajectory enters this module. Input channel d1 is the constant
``drive`` multiplier; channel d2 remains one. Sparse synapses retain the API's
one-step delay, spike amplitude 30, Euler update order and tensile muscle signs.
Use train.py verify to compare histories against the actual API before training.
"""
from copy import deepcopy
import numpy as np
from numba import njit
from simulate import PAPER_CONNECTIONS, validate_system

DT_MS = .1
DT_S = .0001
EDGES = tuple(sorted(PAPER_CONNECTIONS, key=lambda edge: (edge[1], edge[0])))
SOURCES = np.array([e[0] for e in EDGES], dtype=np.int64)
TARGETS = np.array([e[1] for e in EDGES], dtype=np.int64)
PATHS = (
    [('network', 'neuron_params', key, i) for key in 'abcd' for i in range(10)]
    + [('network', 'weights', t, s) for s, t, _ in EDGES]
    + [('network', 'tau_syn', t, s) for s, t, _ in EDGES]
    + [('network', 'Q_aff', i, j) for i in range(10) for j in range(6)]
    + [('network', 'Q_app', i, j) for i in range(10) for j in range(2)]
    + [('limbs', 0, muscle, key) for muscle in ('flexor', 'extensor')
       for key in ('w', 'N', 'A', 'tau_c', 'tau_1')]
    + [('limbs', 0, 'mechanics', key) for key in
       ('mass', 'length', 'viscosity', 'tendon_a1', 'tendon_a2', 'q0', 'w0')]
)
PATH_INDEX = {path: i for i, path in enumerate(PATHS)}
TRACE_COLUMNS = ['time_s', 'q_rad', 'w_rad_s', 'force_flex_N', 'force_ext_N'] + \
    [f'V_{i}' for i in range(10)] + [f'U_{i}' for i in range(10)] + \
    ['Ia_F', 'II_F', 'Ib_F', 'Ia_E', 'II_E', 'Ib_E'] + \
    ['cumulative_RG_F', 'cumulative_RG_E', 'cumulative_Mn_F', 'cumulative_Mn_E']


def pack(config):
    """Validate and pack an API config into 165 physical values."""
    validate_system(config)
    if config['simulation']['dt'] != DT_MS:
        raise ValueError('Fast backend requires the calibrated dt=0.1 ms')
    result = []
    for path in PATHS:
        value = config
        for key in path:
            value = value[key]
        result.append(value)
    return np.asarray(result, dtype=float)


def unpack(physical, template):
    """Recover a full API config, preserving metadata and inactive matrix entries."""
    p = np.asarray(physical)
    if p.shape != (len(PATHS),) or not np.isfinite(p).all():
        raise ValueError('Expected 165 finite parameters')
    result = deepcopy(template)
    for path, value in zip(PATHS, p):
        node = result
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = int(value) if path[-1] == 'N' else float(value)
    validate_system(result)
    return result


@njit(cache=True)
def integrate(p, steps, drive=1., stride=10, detailed=False, angle_limit=np.pi):
    """Native pre-step histories; bail out on nonfinite state or excursion.

    angle_limit is measured from the downward vertical. Training can impose
    a task-specific excursion limit; this is not a hard stop in the mechanics.
    Return trace, completed steps, failure (0 success / 1 excursion / 2 numbers).
    """
    V = p[20:30].copy()
    U = V * p[10:20]
    spike = np.zeros(10)
    syn = np.zeros(14)
    inv_syn = 1. / p[54:68]
    inv_calcium = np.array([1./p[151], 1./p[156]])
    inv_force = np.array([1./p[152], 1./p[157]])
    aff = np.zeros(6)
    cn, force, activation = np.zeros(2), np.zeros(2), np.zeros(2)
    counts = np.zeros(4)
    total = np.zeros(10)
    a1, a2 = p[161], p[162]
    mass, length, viscosity = p[158], p[159], p[160]
    q, w = p[163], p[164]
    inertia = mass * length**2 / 3
    lth = np.sqrt(a1*a1 + a2*a2)
    rows = (steps + stride - 1) // stride
    trace = np.empty((rows, 35 if detailed else 5))
    row = 0
    for step in range(steps):
        if step % stride == 0:
            trace[row, 0] = step * DT_S
            trace[row, 1], trace[row, 2] = q, w
            trace[row, 3], trace[row, 4] = force[0], force[1]
            if detailed:
                trace[row, 5:15] = V
                trace[row, 15:25] = U
                trace[row, 25:31] = aff
                trace[row, 31:35] = counts
            row += 1
        for j, neuron in enumerate((0, 1, 8, 9)):
            counts[j] += V[neuron] >= 30
        for i in range(10):
            total[i] = 0.
        for k in range(14):
            total[TARGETS[k]] += syn[k]
        # All neural derivatives see previous voltages and previous synapses.
        for i in range(10):
            current = p[128+2*i] * drive + p[129+2*i]
            sensory = 0.
            for j in range(6):
                sensory += p[68+6*i+j] * aff[j]
            current += sensory
            old_v, old_u = V[i], U[i]
            dv = .04 * old_v**2 + 5*old_v + 140 - old_u + current + total[i]
            du = p[i] * (p[10+i]*old_v - old_u)
            if old_v >= 30:
                V[i], U[i] = p[20+i], old_u+p[30+i]
            else:
                V[i], U[i] = old_v+DT_MS*dv, old_u+DT_MS*du
        # Synaptic current uses spikes from the PREVIOUS neural step.
        for k in range(14):
            syn[k] += DT_MS * (-syn[k] * inv_syn[k] + p[40+k]*spike[SOURCES[k]])
        for i in range(10):
            spike[i] = 30. if V[i] >= 30 else 0.
            if V[i] >= 30:
                V[i] = 31.
            if not np.isfinite(V[i]) or not np.isfinite(U[i]) or abs(V[i]) > 1e8:
                return trace[:row], step, 2
        for muscle in range(2):
            j = 148 + 5*muscle
            cn[muscle] += DT_MS * (p[j]*spike[8+muscle] - cn[muscle]*inv_calcium[muscle])
            power = cn[muscle]**2.5
            activation[muscle] = power / (power + .75**2.5)
            force[muscle] += DT_MS * (p[j+2]*p[j+1]*activation[muscle] - force[muscle]*inv_force[muscle])
        lf = np.sqrt(a1*a1+a2*a2-2*a1*a2*np.cos(q))
        le = np.sqrt(a1*a1+a2*a2-2*a1*a2*np.cos(np.pi-q))
        hf = a1*a2*np.sin(q)/lf
        he = a1*a2*np.sin(np.pi-q)/le
        torque = -force[0]*hf + force[1]*he
        next_w = w + DT_S * (.5*9.81*mass*length*np.cos(q) - viscosity*w + torque)/inertia
        q += DT_S*w
        w = next_w
        if not np.isfinite(q) or not np.isfinite(w) or not np.isfinite(force).all():
            return trace[:row], step+1, 2
        if abs(q-np.pi/2) > angle_limit:
            return trace[:row], step+1, 1
        # New afferents are computed AFTER muscle and mechanical updates.
        lf = np.sqrt(a1*a1+a2*a2-2*a1*a2*np.cos(q))
        le = np.sqrt(a1*a1+a2*a2-2*a1*a2*np.cos(np.pi-q))
        for muscle in range(2):
            lm = lf if muscle == 0 else le
            vm = a1*a2*np.sin(q)*w/lf if muscle == 0 else -a1*a2*np.sin(np.pi-q)*w/le
            vnorm = max(vm/lth, 0.)
            dnorm = max((lm-lth)/lth, 0.)
            aff[3*muscle] = 6.2*vnorm**.6 + 2*dnorm + .06*activation[muscle]
            aff[3*muscle+1] = 1.5*dnorm + .06*activation[muscle]
            aff[3*muscle+2] = max((force[muscle]-3.38)/3.38, 0.)
    return trace, steps, 0


def replay(physical, *, duration_s=10., drive=1., stride=10, detailed=True, angle_limit_deg=180.):
    """Checked Python entry point, seconds at the interface, fresh state per run."""
    if (not np.isfinite([duration_s, drive, angle_limit_deg]).all() or duration_s <= 0
            or drive < 0 or not isinstance(stride, int) or stride < 1 or angle_limit_deg <= 0
            or not np.isclose(duration_s/DT_S, round(duration_s/DT_S), rtol=0, atol=1e-7)):
        raise ValueError('Invalid duration, drive, stride or angle limit')
    p = np.ascontiguousarray(physical, dtype=float)
    if p.shape != (165,) or not np.isfinite(p).all():
        raise ValueError('Expected 165 finite physical parameters')
    return integrate(p, round(duration_s/DT_S), drive, stride, detailed, np.deg2rad(angle_limit_deg))
