"""Shared full-Afferents dynamics, explicit mechanics and initial-condition banks.

This module has one Numba integrator and one independent library replay. Inputs
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
    kick_step = round(case[3] * 1000 / DT_MS) if case[3] >= 0 else -1
    for step in range(round(duration_s * 1000 / DT_MS)):
        if step == kick_step:
            limb.w += case[4]
            limb.w_prev = limb.w
        net.step(dt=DT_MS, Iapp=np.asarray(currents), Iaff=aff)
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
def _integrate(p, case, currents, mechanics, receptors, duration_s, stride):
    """Integrate fixed-step neural dynamics; all changing physical inputs are explicit."""
    dt_ms = DT_MS
    MASS, LENGTH, VISCOSITY, GRAVITY = mechanics
    A1, A2 = p[46], p[47]
    INERTIA = MASS * LENGTH**2 / 3
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
    q, omega = TARGET + case[0], case[1]
    v, u = p[4:6].copy(), p[4:6] * p[2:4]
    syn = np.zeros((2, 2))
    output = np.zeros(2)
    new_output = np.zeros(2)
    calcium, force, aff = np.zeros(2), np.zeros(2), np.zeros(6)
    activation = np.zeros(2)
    if stride:
        trace[0, 1:3] = case[:2]
        trace[0, 5:7], trace[0, 7:9] = v, u
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
            omega += case[3]
        for i in range(2):
            feedback = 0.
            for j in range(6):
                feedback += p[30+6*i+j] * aff[j]
            current = currents[i] + feedback
            if v[i] >= 30:
                new_v, new_u = p[4+i], u[i] + p[6+i]
            else:
                new_v = v[i] + dt_ms * (.04*v[i]**2 + 5*v[i] + 140 - u[i] + current + (syn[i, 0] + syn[i, 1]))
                new_u = u[i] + dt_ms * p[i] * (p[2+i]*v[i] - u[i])
            for j in range(2):
                syn[i, j] += dt_ms * (-syn[i, j]*(1/p[12+2*i+j]) + p[8+2*i+j]*output[j])
            v[i], u[i] = 31. if new_v >= 30 else new_v, new_u
            new_output[i] = 30. if new_v >= 30 else 0.
        output, new_output = new_output, output
        for i in range(2):
            k = 16 + 7*i
            drive = p[48+2*i]*output[0] + p[49+2*i]*output[1]
            calcium[i] += dt_ms * (p[k]*drive - calcium[i]*(1/p[k+3]))
            activation[i] = calcium[i]**p[k+5] / (calcium[i]**p[k+5] + p[k+6]**p[k+5])
            force[i] += dt_ms * (p[k+2]*p[k+1]*activation[i] - force[i]*(1/p[k+4]))

        lf = np.sqrt(A1*A1 + A2*A2 - 2*A1*A2*np.cos(q))
        le = np.sqrt(A1*A1 + A2*A2 - 2*A1*A2*np.cos(np.pi-q))
        hf, he = A1*A2*np.sin(q)/lf, A1*A2*np.sin(np.pi-q)/le
        # Positive tension shortens a tendon: M_i = -F_i * dL_i/dq.
        torque = -force[0]*hf + force[1]*he
        next_omega = omega + .001*dt_ms * (.5*GRAVITY*MASS*LENGTH*np.cos(q) - VISCOSITY*omega + torque)/INERTIA
        q += .001*dt_ms * omega
        omega = next_omega
        # Same post-mechanical-step afferents and one-step feedback delay as core.
        lf = np.sqrt(A1*A1 + A2*A2 - 2*A1*A2*np.cos(q))
        le = np.sqrt(A1*A1 + A2*A2 - 2*A1*A2*np.cos(np.pi-q))
        vf = A1*A2*np.sin(q)/lf * omega
        ve = -A1*A2*np.sin(np.pi-q)/le * omega
        for i in range(2):
            length, velocity = (lf, vf) if i == 0 else (le, ve)
            aff[3*i], aff[3*i+1], aff[3*i+2] = afferent_values(
                velocity, length, force[i], activation[i], p[42+i], p[44+i], receptors)
        theta = q - TARGET
        completed = step + 1
        finite = np.isfinite(theta) and np.isfinite(omega)
        for i in range(2):
            finite = finite and np.isfinite(v[i]) and np.isfinite(u[i]) and np.isfinite(force[i]) and np.isfinite(calcium[i])
        for i in range(6):
            finite = finite and np.isfinite(aff[i])
        if not finite:
            return np.array([-1., completed*dt_ms/1000, angle_limit, angle_limit, 0., 0., duration_s, angle_limit, angle_limit]), trace[:row]
        error = theta-target_rad
        error2 += error*error
        peak = max(peak, abs(error))
        effort += force[0] + force[1]
        if abs(error) > SETTLE_ANGLE or abs(omega) > SETTLE_SPEED:
            last_outside = completed*dt_ms/1000
        reward += max(0., np.exp(-(error/np.deg2rad(15.))**2) - .05*min(omega*omega, 4.))
        if step >= late_start:
            late_quality += np.exp(-(error/np.deg2rad(3.))**2)
            late_error2 += error*error
            late_count += 1
        if stride and (step % stride == 0 or step == steps-1 or abs(theta) > angle_limit):
            trace[row, 0:5] = np.array([completed*dt_ms/1000, theta, omega, force[0], force[1]])
            trace[row, 5:7], trace[row, 7:9] = v, u
            trace[row, 9:11], trace[row, 11:17] = calcium, aff
            trace[row, 17:19] = activation
            trace[row, 19:21] = output
            trace[row, 21:25] = syn.reshape(4)
            trace[row, 25] = torque
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
