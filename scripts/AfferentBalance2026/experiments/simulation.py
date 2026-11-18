"""Shared full-Afferents dynamics, explicit mechanics and initial-condition banks.

All episode and streaming integrators share one Numba physical step. Inputs
are the 52-field genome and two external currents. There is no PD controller.
"""
import numpy as np
from numba import njit
from SpikingNN.core.Izh_net import Afferents
from .core.genome import GenomeConfig

DT_MS = .1
MODEL = "SpikingNN.Afferents/v3-tensile"
TARGET = -np.pi / 2
ANGLE_LIMIT = np.pi / 2
SETTLE_ANGLE = np.deg2rad(3.)
SETTLE_SPEED = .35
SETTLE_DEADLINE = 3.
CASE_COLUMNS = ('target_rad', 'theta0_rad', 'omega0_rad_s', 'kick_time_s', 'kick_delta_omega')
METRIC_COLUMNS = ('score', 'survival_s', 'rmse_rad', 'peak_error_rad', 'mean_force_N',
                  'success', 'settling_s', 'late_rmse_rad', 'final_theta_rad')
TRACE_COLUMNS = ('time_s', 'theta_rad', 'omega_rad_s', 'force_flex_N', 'force_ext_N',
                 'V_flex', 'V_ext', 'U_flex', 'U_ext', 'calcium_flex', 'calcium_ext',
                 'Ia_F', 'II_F', 'Ib_F', 'Ia_E', 'II_E', 'Ib_E',
                 'activation_flex', 'activation_ext', 'spike_flex', 'spike_ext',
                 'syn_00', 'syn_01', 'syn_10', 'syn_11', 'applied_torque_Nm')
PARAMETER_NAMES = ('p_v', 'k_v', 'k_dI', 'k_dII', 'k_nI', 'k_nII', 'k_f', 'const_I', 'const_II')
PARAMETERS = tuple(float(getattr(Afferents, name)) for name in PARAMETER_NAMES)


def simulate(physical, case, mechanics, *, currents=(0., 0.), duration_s=10., stride=1):
    """Return episode metrics and a trace including the reset state at t=0.

    Parameters
    ----------
    physical : array_like, shape (52,)
        Physical genome in the new schema; indices 46/47 contain a1/a2.
    case : array_like, shape (5,)
        Target, initial angle, speed, kick time and velocity kick, in SI units.
        The target is used only for scoring, never as a neural feedback signal.
    mechanics : array_like, shape (4,)
        Mass, link length, viscosity, gravity from GenomeConfig.mechanics.
    currents : array_like, shape (2,)
        Constant external neural currents; exactly zero in balance training.
    duration_s : float
        Positive duration on the fixed 0.1 ms grid.
    stride : int
        Record every step for 1; 0 computes metrics without a trace.
    """
    p = GenomeConfig.validate_physical(physical)
    case, mechanics, currents = (np.asarray(x, dtype=float) for x in (case, mechanics, currents))
    if case.shape != (5,) or not np.isfinite(case).all() or np.any(np.abs(case[:2]) >= ANGLE_LIMIT):
        raise ValueError("Expected a finite case with target and initial angle inside ±90 degrees")
    if (mechanics.shape != (4,) or not np.isfinite(mechanics).all()
            or np.any(mechanics[[0, 1, 3]] <= 0) or mechanics[2] < 0):
        raise ValueError("Invalid mechanics")
    if currents.shape != (2,) or not np.isfinite(currents).all():
        raise ValueError("Expected two finite external currents")
    if (not np.isfinite(duration_s) or duration_s < .0001
            or not np.isclose(duration_s * 10000, round(duration_s * 10000), rtol=0, atol=1e-7)
            or not isinstance(stride, (int, np.integer)) or stride < 0):
        raise ValueError("Use a positive duration on the 0.1 ms grid and a nonnegative integer stride")
    return _integrate(p, case, currents, mechanics, PARAMETERS, duration_s, stride)


def case_grid(angles_deg=None, speeds=None, target_deg=0.):
    """Build an absolute-angle Cartesian grid, with no external perturbation."""
    angles = np.asarray(np.linspace(-50., 50., 10) if angles_deg is None else angles_deg, dtype=float)
    speeds = np.asarray(np.linspace(-1., 1., 10) if speeds is None else speeds, dtype=float)
    if (angles.ndim != 1 or speeds.ndim != 1 or not len(angles) or not len(speeds)
            or not np.isfinite(angles).all() or not np.isfinite(speeds).all()
            or np.any(abs(angles) >= 90) or not np.isfinite(target_deg) or abs(target_deg) >= 90):
        raise ValueError("Expected finite nonempty angle/speed vectors inside ±90 degrees")
    return np.array([[np.deg2rad(target_deg), np.deg2rad(angle), speed, -1., 0.]
                     for speed in speeds for angle in angles])


def balance_banks(settings):
    """Sample the complete configured angle/speed rectangle without a central gap.

    Independent seeded Latin hypercubes cover the training, validation and test
    domains. Training additionally includes both angular endpoints and the centre
    at the midpoint speed. All conditions remain fixed throughout a GA run.
    Configured ranges may be asymmetric; no mirrored-pair restriction is imposed.
    """
    domain = settings["initial_conditions"]
    bounds = np.array([domain["angle_range_deg"], domain["speed_range_rad_s"]], dtype=float)
    if (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1]) or np.any(abs(bounds[0]) >= 90)):
        raise ValueError("Expected increasing finite angle/speed ranges inside ±90 degrees")
    seeds = settings["bank_seeds"]
    if set(seeds) != {"training", "validation", "test"} or len(set(seeds.values())) != 3:
        raise ValueError("Training, validation and test must use independent seeds")
    result = {}
    for name in ("training", "validation", "test"):
        count = settings[f"{name}_count"]
        if not isinstance(count, int) or count < 4:
            raise ValueError("Bank counts must be integers of at least four")
        rng = np.random.default_rng(seeds[name])
        anchors = 3 if name == "training" else 0
        sampled = count - anchors
        unit = np.column_stack([(rng.permutation(sampled) + rng.random(sampled)) / sampled
                                for _ in range(2)])
        coordinates = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
        if anchors:
            midpoint = bounds.mean(axis=1)
            fixed = np.array([[bounds[0, 0], midpoint[1]], midpoint,
                              [bounds[0, 1], midpoint[1]]])
            coordinates = np.concatenate((fixed, coordinates))
        cases = np.zeros((count, 5))
        cases[:, 1] = np.deg2rad(coordinates[:, 0])
        cases[:, 2] = coordinates[:, 1]
        cases[:, 3] = -1.
        result[name] = cases
    return result


def reference_trace(physical, case, mechanics, *, currents=(0., 0.), duration_s=.05):
    """Independently replay all recorded channels through original library classes.

    Return a full-resolution trace including t=0. This intentionally duplicates
    the update sequence for verification instead of calling the Numba integrator.
    """
    from SpikingNN.core.Izh_net import Izhikevich_IO_Network, OneDOFLimb, SimpleAdaptedMuscle
    p = GenomeConfig.validate_physical(physical)
    net = Izhikevich_IO_Network(
        N=2, input_size=2, output_size=2, afferent_size=6,
        a=p[:2], b=p[2:4], c=p[4:6], d=p[6:8], W=p[8:12].reshape(2, 2),
        tau_syn=p[12:16].reshape(2, 2), Q_app=np.eye(2),
        Q_aff=p[30:42].reshape(2, 6), P=p[48:52].reshape(2, 2))
    muscles = [SimpleAdaptedMuscle(**dict(zip(
        ("w", "N", "A", "tau_c", "tau_1", "m", "k"), p[16+7*i:23+7*i])))
        for i in range(2)]
    mass, length, viscosity, gravity = mechanics
    limb = OneDOFLimb(m=mass, ls=length, b=viscosity, a1=p[46], a2=p[47],
                      q0=TARGET+case[1], w0=case[2])
    limb.g = gravity
    receptors = [Afferents(), Afferents()]
    for i, receptor in enumerate(receptors):
        receptor.L_th, receptor.F_th = p[42+i], p[44+i]
    aff = np.zeros(6)
    reset = np.zeros(26)
    reset[1:3], reset[5:7], reset[7:9] = case[1:3], p[4:6], p[4:6]*p[2:4]
    history = [reset]
    steps = round(duration_s * 1000 / DT_MS)
    currents = np.asarray(currents, dtype=float)
    if currents.shape not in ((2,), (steps, 2)) or not np.isfinite(currents).all():
        raise ValueError('Expected two constant currents or one pair per integration interval')
    kick_step = round(case[3] * 1000 / DT_MS) if case[3] >= 0 else -1
    for step in range(steps):
        if step == kick_step:
            limb.w += case[4]
            limb.w_prev = limb.w
        net.step(dt=DT_MS, Iapp=currents if currents.ndim == 1 else currents[step], Iaff=aff)
        for i, muscle in enumerate(muscles):
            muscle.step(dt=DT_MS, u=net.V_out[i])
        limb.step(dt=DT_MS, F_flex=muscles[0].F, F_ext=muscles[1].F)
        for i, receptor in enumerate(receptors):
            angle = limb.q if i == 0 else np.pi-limb.q
            size = limb.L(angle)
            velocity = limb.h(size, angle) * limb.w * (1 if i == 0 else -1)
            aff[3*i:3*i+3] = [receptor.Ia(velocity, size, muscles[i].x),
                              receptor.II(size, muscles[i].x), receptor.Ib(muscles[i].F)]
        history.append(np.array([
            (step+1)*DT_MS/1000, limb.q-TARGET, limb.w,
            muscles[0].F, muscles[1].F, *net.V_prev, *net.U_prev,
            muscles[0].Cn, muscles[1].Cn, *aff, muscles[0].x, muscles[1].x,
            *net.output, *net.I_syn.ravel(), limb.M_tot]))
        if not np.isfinite(history[-1]).all() or abs(limb.q-TARGET) > ANGLE_LIMIT:
            break
    return np.asarray(history)


@njit(cache=True, inline="always")
def afferent_values(velocity, length, force, activation, length_threshold,
                    force_threshold, parameters):
    """Evaluate all nonlinear Ia, II and Ib terms for one muscle.

    Parameters
    ----------
    velocity, length, force : float
        Muscle lengthening speed (m/s), length (m), and force (N).
    activation : float
        Hill activation x, not the raw calcium concentration.
    length_threshold, force_threshold : float
        Positive normalization thresholds L_th (m) and F_th (N).
    parameters : tuple of float
        Fixed receptor constants in PARAMETER_NAMES order.

    Returns
    -------
    tuple of float
        Ia, II and Ib responses in library model units. Negative normalized
        speed, stretch and force excess are rectified before evaluation.
    """
    p_v, k_v, k_dI, k_dII, k_nI, k_nII, k_f, const_I, const_II = parameters
    v = max(velocity / length_threshold, 0.)
    d = max((length - length_threshold) / length_threshold, 0.)
    f = max((force - force_threshold) / force_threshold, 0.)
    return (k_v * v**p_v + k_dI * d + k_nI * activation + const_I,
            k_dII * d + k_nII * activation + const_II, k_f * f)


@njit(cache=True)
def initial_state(p, theta_rad, omega_rad_s):
    """Reset the complete physical state without discarding hidden neural variables.

    State layout: q=0, omega=1, V=2:4, U=4:6, synapses=6:10,
    spikes=10:12, calcium=12:14, forces=14:16, afferents=16:22,
    activation=22:24, torque=24. The internal q includes the library offset.
    """
    state = np.zeros(25)
    state[0], state[1] = TARGET + theta_rad, omega_rad_s
    state[2:4], state[4:6] = p[4:6], p[4:6] * p[2:4]
    return state


@njit(cache=True, inline="always")
def physical_step(p, mechanics, receptors, state, currents):
    """Advance the complete state in place by 0.1 ms using two applied currents.

    Both spike outputs remain from the previous step until both neurons have
    updated. Afferents are evaluated after mechanics and feed the following
    neural step, matching the original library update order.
    """
    dt_ms = DT_MS
    mass, length, viscosity, gravity = mechanics
    a1, a2 = p[46], p[47]
    inertia = mass * length**2 / 3
    output0, output1 = 0., 0.
    for i in range(2):
        feedback = 0.
        for j in range(6):
            feedback += p[30+6*i+j] * state[16+j]
        current = currents[i] + feedback
        v, u = state[2+i], state[4+i]
        if v >= 30:
            new_v, new_u = p[4+i], u + p[6+i]
        else:
            new_v = v + dt_ms * (.04*v**2 + 5*v + 140 - u + current + (state[6+2*i] + state[7+2*i]))
            new_u = u + dt_ms * p[i] * (p[2+i]*v - u)
        for j in range(2):
            index = 6+2*i+j
            state[index] += dt_ms * (-state[index]*(1/p[12+2*i+j]) + p[8+2*i+j]*state[10+j])
        state[2+i], state[4+i] = 31. if new_v >= 30 else new_v, new_u
        if i == 0:
            output0 = 30. if new_v >= 30 else 0.
        else:
            output1 = 30. if new_v >= 30 else 0.
    state[10], state[11] = output0, output1
    for i in range(2):
        k = 16+7*i
        drive = p[48+2*i]*state[10] + p[49+2*i]*state[11]
        state[12+i] += dt_ms * (p[k]*drive - state[12+i]*(1/p[k+3]))
        state[22+i] = state[12+i]**p[k+5] / (state[12+i]**p[k+5] + p[k+6]**p[k+5])
        state[14+i] += dt_ms * (p[k+2]*p[k+1]*state[22+i] - state[14+i]*(1/p[k+4]))
    q, omega = state[0], state[1]
    lf = np.sqrt(a1*a1 + a2*a2 - 2*a1*a2*np.cos(q))
    le = np.sqrt(a1*a1 + a2*a2 - 2*a1*a2*np.cos(np.pi-q))
    hf, he = a1*a2*np.sin(q)/lf, a1*a2*np.sin(np.pi-q)/le
    # Positive tension shortens a tendon: M_i = -F_i * dL_i/dq.
    torque = -state[14]*hf + state[15]*he
    next_omega = omega + .001*dt_ms * (.5*gravity*mass*length*np.cos(q) - viscosity*omega + torque)/inertia
    q += .001*dt_ms * omega
    omega = next_omega
    state[0], state[1], state[24] = q, omega, torque
    lf = np.sqrt(a1*a1 + a2*a2 - 2*a1*a2*np.cos(q))
    le = np.sqrt(a1*a1 + a2*a2 - 2*a1*a2*np.cos(np.pi-q))
    vf = a1*a2*np.sin(q)/lf * omega
    ve = -a1*a2*np.sin(np.pi-q)/le * omega
    for i in range(2):
        size, velocity = (lf, vf) if i == 0 else (le, ve)
        state[16+3*i], state[17+3*i], state[18+3*i] = afferent_values(
            velocity, size, state[14+i], state[22+i], p[42+i], p[44+i], receptors)


@njit(cache=True, inline="always")
def write_state(row, state, time_s):
    """Write one state into the existing public 26-column trace layout."""
    row[0], row[1], row[2] = time_s, state[0]-TARGET, state[1]
    row[3:5], row[5:7], row[7:9] = state[14:16], state[2:4], state[4:6]
    row[9:11], row[11:17], row[17:19] = state[12:14], state[16:22], state[22:24]
    row[19:21], row[21:25], row[25] = state[10:12], state[6:10], state[24]


@njit(cache=True)
def _integrate(p, case, currents, mechanics, receptors, duration_s, stride):
    """Integrate fixed-step neural dynamics; all changing physical inputs are explicit."""
    dt_ms = DT_MS
    target_rad = case[0]
    case = case[1:]
    angle_limit = ANGLE_LIMIT
    if not np.isfinite(duration_s) or duration_s < DT_MS / 1000:
        raise ValueError("duration_s must include at least one step")
    if dt_ms != DT_MS:
        raise ValueError("All experiments require dt_ms=0.1")
    steps = int(round(duration_s * 1000 / dt_ms))
    trace = np.zeros(((steps + max(stride, 1) - 1) // max(stride, 1) + 2 if stride else 0, 26))
    row = 1 if stride else 0
    state = initial_state(p, case[0], case[1])
    if stride:
        trace[0, 1:3] = case[:2]
        trace[0, 5:7], trace[0, 7:9] = state[2:4], state[4:6]
    reward, error2, peak, effort = 0., 0., abs(case[0]-target_rad), 0.
    kick_step = int(round(case[2] * 1000 / dt_ms)) if case[2] >= 0 else -1
    completed = 0
    late_quality, late_error2, late_count = 0., 0., 0
    last_outside = 0.
    late_start = int(round(max(0., duration_s-2.)*1000/dt_ms))
    late_steps = max(1, steps-late_start)
    theta = case[0]
    for step in range(steps):
        # A kick changes velocity before this step; it is external to the controller.
        if step == kick_step:
            state[1] += case[3]
        physical_step(p, mechanics, receptors, state, currents)
        theta, omega = state[0]-TARGET, state[1]
        completed = step + 1
        finite = np.isfinite(theta) and np.isfinite(omega)
        for i in range(2):
            finite = finite and np.isfinite(state[2+i]) and np.isfinite(state[4+i]) and np.isfinite(state[14+i]) and np.isfinite(state[12+i])
        for i in range(6):
            finite = finite and np.isfinite(state[16+i])
        if not finite:
            return np.array([-1., completed*dt_ms/1000, angle_limit, angle_limit, 0., 0., duration_s, angle_limit, angle_limit]), trace[:row]
        error = theta-target_rad
        error2 += error*error
        peak = max(peak, abs(error))
        effort += state[14] + state[15]
        if abs(error) > SETTLE_ANGLE or abs(omega) > SETTLE_SPEED:
            last_outside = completed*dt_ms/1000
        reward += max(0., np.exp(-(error/np.deg2rad(15.))**2) - .05*min(omega*omega, 4.))
        if step >= late_start:
            late_quality += np.exp(-(error/np.deg2rad(3.))**2)
            late_error2 += error*error
            late_count += 1
        if stride and (step % stride == 0 or step == steps-1 or abs(theta) > angle_limit):
            write_state(trace[row], state, completed*dt_ms/1000)
            row += 1
        if abs(theta) > angle_limit:
            break
    survived = completed == steps and abs(theta) <= angle_limit
    success = 1. if survived and last_outside <= min(SETTLE_DEADLINE, duration_s-2.) else 0.
    score = .10*completed/steps + .30*reward/steps + .45*late_quality/late_steps + .15*success
    late_rmse = np.sqrt(late_error2/late_count) if survived and late_count else angle_limit
    return np.array([score, completed*dt_ms/1000, np.sqrt(error2/completed), peak,
                     effort/completed, success, last_outside if survived else duration_s,
                     late_rmse, theta]), trace[:row]



def measure_trace(trace, target_rad=0., duration_s=10., dt_ms=DT_MS, angle_limit=ANGLE_LIMIT, initial_theta=None):
    """Recompute nine episode metrics from a full-resolution post-step trace.

    Exclude any prepended reset row. Angles are radians, duration is seconds,
    and dt_ms must equal 0.1. initial_theta optionally includes the reset angle
    in the peak-error metric. Early falls receive the documented late-error
    penalty, not an extrapolated stationary error.
    """
    if dt_ms != DT_MS:
        raise ValueError("All experiments require dt_ms=0.1")
    steps = round(duration_s*1000/dt_ms)
    if not len(trace) or not np.isfinite(trace).all():
        return np.array([-1., len(trace)*dt_ms/1000, angle_limit, angle_limit, 0.,
                         0., duration_s, angle_limit, angle_limit])
    theta, omega = trace[:, 1], trace[:, 2]
    error = theta-target_rad
    survived = len(trace) == steps and abs(theta[-1]) <= angle_limit
    outside = (np.abs(error) > SETTLE_ANGLE) | (np.abs(omega) > SETTLE_SPEED)
    last_outside = trace[outside, 0][-1] if outside.any() else 0.
    success = float(survived and last_outside <= min(SETTLE_DEADLINE, duration_s-2.))
    late_start = round(max(0., duration_s-2.)*1000/dt_ms)
    late = error[late_start:]
    quality = np.maximum(0., np.exp(-(error/np.deg2rad(15.))**2)-.05*np.minimum(omega**2,4.))
    score = (.10*len(trace)/steps + .30*quality.sum()/steps
             + .45*np.exp(-(late/np.deg2rad(3.))**2).sum()/max(1,steps-late_start) + .15*success)
    return np.array([score, len(trace)*dt_ms/1000, np.sqrt(np.mean(error**2)),
                     max(np.max(np.abs(error)), abs(initial_theta-target_rad) if initial_theta is not None else 0.), np.mean(trace[:,3]+trace[:,4]), success,
                     last_outside if survived else duration_s,
                     np.sqrt(np.mean(late**2)) if survived and len(late) else angle_limit,
                     theta[-1]])
