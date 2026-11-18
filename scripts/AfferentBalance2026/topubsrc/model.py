"""Standalone full-Afferents dynamics, explicit mechanics and episode fitness.

All episode and streaming integrators share one Numba physical step. Inputs
are the 52-field genome and two external currents. There is no PD controller.
"""
import numpy as np
from numba import njit, prange
from genome import GenomeConfig

DT_MS = .1
MODEL = "SpikingNN.Afferents/v3-tensile"
TARGET = -np.pi / 2
ANGLE_LIMIT = np.pi / 2
SETTLE_ANGLE = np.deg2rad(3.)
SETTLE_SPEED = .35
SETTLE_DEADLINE = 3.
DEFAULT_FITNESS_WINDOW_S = 2.
CASE_COLUMNS = ('target_rad', 'theta0_rad', 'omega0_rad_s', 'kick_time_s', 'kick_delta_omega')
METRIC_COLUMNS = ('score', 'survival_s', 'rmse_rad', 'peak_error_rad', 'mean_force_N',
                  'success', 'settling_s', 'late_rmse_rad', 'final_theta_rad')
TRACE_COLUMNS = ('time_s', 'theta_rad', 'omega_rad_s', 'force_flex_N', 'force_ext_N',
                 'V_flex', 'V_ext', 'U_flex', 'U_ext', 'calcium_flex', 'calcium_ext',
                 'Ia_F', 'II_F', 'Ib_F', 'Ia_E', 'II_E', 'Ib_E',
                 'activation_flex', 'activation_ext', 'spike_flex', 'spike_ext',
                 'syn_00', 'syn_01', 'syn_10', 'syn_11', 'applied_torque_Nm')
PARAMETER_NAMES = ('p_v', 'k_v', 'k_dI', 'k_dII', 'k_nI', 'k_nII', 'k_f', 'const_I', 'const_II')
PARAMETERS = (0.6, 6.2, 2.0, 1.5, 0.06, 0.06, 1.0, 0.0, 0.0)
# Frozen receptor constants; no external neural-network library is imported.


def validate_duration(duration_s):
    """Reject invalid episode lengths before allocating output or compiling Numba."""
    if (not isinstance(duration_s, (int, float, np.integer, np.floating))
            or not np.isfinite(duration_s) or duration_s < .0001
            or not np.isclose(duration_s * 10000, round(duration_s * 10000), rtol=0, atol=1e-7)):
        raise ValueError('Duration must be positive and lie on the 0.1 ms grid')


def validate_fitness_window(window_s):
    """Validate the final accuracy window; older genomes use two seconds."""
    try:
        validate_duration(window_s)
    except ValueError as exc:
        raise ValueError('fitness_window_s must be positive and lie on the 0.1 ms grid') from exc
    return float(window_s)


def firing_interval_steps(max_firing_rate_hz=None):
    """Convert an optional instantaneous rate limit to the fixed-step ISI bound.

    None preserves the historical unconstrained score. A 100 Hz limit means
    at least 100 integration steps (10 ms) between successive spikes of EACH
    neuron, including transients. This changes admissibility, not dynamics.
    """
    if max_firing_rate_hz is None:
        return 0
    if (isinstance(max_firing_rate_hz, (bool, np.bool_))
            or not isinstance(max_firing_rate_hz, (int, float, np.integer, np.floating))
            or not np.isfinite(max_firing_rate_hz) or not 0 < max_firing_rate_hz <= 1000/DT_MS):
        raise ValueError('max_firing_rate_hz must be positive and finite, or null to disable')
    return int(np.ceil(1000/(DT_MS*max_firing_rate_hz)-1e-12))


@njit(cache=True)
def constrained_score(score, success, shortest_interval, minimum_interval):
    """Feasible episodes retain the old score; a rate violation scores below -1.

    The negative score orders infeasible candidates by their worst ISI ratio.
    Bank fitness treats any such episode as a hard feasibility violation.
    """
    if minimum_interval and shortest_interval < minimum_interval:
        return -float(minimum_interval)/shortest_interval, 0.
    return score, success


def simulate(physical, case, mechanics, *, currents=(0., 0.), duration_s=10., stride=1,
             fitness_window_s=DEFAULT_FITNESS_WINDOW_S, max_firing_rate_hz=None):
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
    fitness_window_s : float
        Final interval for accuracy reward and late RMSE. Strict success also
        requires settling by min(3, duration_s - fitness_window_s) seconds.
    max_firing_rate_hz : float or None
        Optional hard ISI constraint on each neuron over the whole episode.
        Rate-violating trajectories are recorded unchanged, but their score
        is negative and strict success is zero. None retains historical scores.
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
    validate_duration(duration_s)
    fitness_window_s = validate_fitness_window(fitness_window_s)
    if not isinstance(stride, (int, np.integer)) or stride < 0:
        raise ValueError("Use a nonnegative integer stride")
    interval = firing_interval_steps(max_firing_rate_hz)
    return _integrate(p, case, currents, mechanics, PARAMETERS, duration_s, stride, fitness_window_s, interval)


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
def _integrate(p, case, currents, mechanics, receptors, duration_s, stride,
               fitness_window_s=DEFAULT_FITNESS_WINDOW_S, minimum_interval=0):
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
    late_start = int(round(max(0., duration_s-fitness_window_s)*1000/dt_ms))
    late_steps = max(1, steps-late_start)
    theta = case[0]
    last_spike = np.full(2, -1, dtype=np.int64)
    shortest_interval = 2**62
    for step in range(steps):
        # A kick changes velocity before this step; it is external to the controller.
        if step == kick_step:
            state[1] += case[3]
        physical_step(p, mechanics, receptors, state, currents)
        if minimum_interval:
            for neuron in range(2):
                if state[10+neuron] > 0:
                    if last_spike[neuron] >= 0:
                        shortest_interval = min(shortest_interval, step-last_spike[neuron])
                    last_spike[neuron] = step
        theta, omega = state[0]-TARGET, state[1]
        completed = step + 1
        finite = np.isfinite(theta) and np.isfinite(omega)
        for i in range(2):
            finite = finite and np.isfinite(state[2+i]) and np.isfinite(state[4+i]) and np.isfinite(state[14+i]) and np.isfinite(state[12+i])
        for i in range(6):
            finite = finite and np.isfinite(state[16+i])
        if not finite:
            return np.array([-1e6 if minimum_interval else -1., completed*dt_ms/1000, angle_limit, angle_limit, 0., 0., duration_s, angle_limit, angle_limit]), trace[:row]
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
    success = 1. if survived and last_outside <= min(SETTLE_DEADLINE, duration_s-fitness_window_s) else 0.
    score = .10*completed/steps + .30*reward/steps + .45*late_quality/late_steps + .15*success
    score, success = constrained_score(score, success, shortest_interval, minimum_interval)
    late_rmse = np.sqrt(late_error2/late_count) if survived and late_count else angle_limit
    return np.array([score, completed*dt_ms/1000, np.sqrt(error2/completed), peak,
                     effort/completed, success, last_outside if survived else duration_s,
                     late_rmse, theta]), trace[:row]



def measure_trace(trace, target_rad=0., duration_s=10., dt_ms=DT_MS, angle_limit=ANGLE_LIMIT,
                  initial_theta=None, fitness_window_s=DEFAULT_FITNESS_WINDOW_S, max_firing_rate_hz=None):
    """Recompute nine episode metrics from a full-resolution post-step trace.

    Exclude any prepended reset row. Angles are radians, duration is seconds,
    and dt_ms must equal 0.1. initial_theta optionally includes the reset angle
    in the peak-error metric. Early falls receive the documented late-error
    penalty, not an extrapolated stationary error.
    """
    if dt_ms != DT_MS:
        raise ValueError("All experiments require dt_ms=0.1")
    validate_duration(duration_s)
    fitness_window_s = validate_fitness_window(fitness_window_s)
    minimum_interval = firing_interval_steps(max_firing_rate_hz)
    steps = round(duration_s*1000/dt_ms)
    if not len(trace) or not np.isfinite(trace).all():
        return np.array([-1e6 if minimum_interval else -1., len(trace)*dt_ms/1000, angle_limit, angle_limit, 0.,
                         0., duration_s, angle_limit, angle_limit])
    theta, omega = trace[:, 1], trace[:, 2]
    error = theta-target_rad
    survived = len(trace) == steps and abs(theta[-1]) <= angle_limit
    outside = (np.abs(error) > SETTLE_ANGLE) | (np.abs(omega) > SETTLE_SPEED)
    last_outside = trace[outside, 0][-1] if outside.any() else 0.
    success = float(survived and last_outside <= min(SETTLE_DEADLINE, duration_s-fitness_window_s))
    late_start = round(max(0., duration_s-fitness_window_s)*1000/dt_ms)
    late = error[late_start:]
    quality = np.maximum(0., np.exp(-(error/np.deg2rad(15.))**2)-.05*np.minimum(omega**2,4.))
    score = (.10*len(trace)/steps + .30*quality.sum()/steps
             + .45*np.exp(-(late/np.deg2rad(3.))**2).sum()/max(1,steps-late_start) + .15*success)
    shortest_interval = 2**62
    if minimum_interval:
        if not np.allclose(np.diff(trace[:, 0]), DT_MS/1000, atol=1e-10, rtol=0):
            raise ValueError('Rate-constrained metrics require stride=1')
        for neuron in range(2):
            spikes = np.flatnonzero(trace[:, 19+neuron] > 0)
            if len(spikes) > 1:
                shortest_interval = min(shortest_interval, int(np.diff(spikes).min()))
    score, success = constrained_score(score, success, shortest_interval, minimum_interval)
    return np.array([score, len(trace)*dt_ms/1000, np.sqrt(np.mean(error**2)),
                     max(np.max(np.abs(error)), abs(initial_theta-target_rad) if initial_theta is not None else 0.), np.mean(trace[:,3]+trace[:,4]), success,
                     last_outside if survived else duration_s,
                     np.sqrt(np.mean(late**2)) if survived and len(late) else angle_limit,
                     theta[-1]])




def fitness(metrics):
    """Reduce a nonempty K-by-9 episode-metric matrix to one robust fitness.

    A score below -1 encodes a rate violation (or a constrained numerical
    failure): the entire bank is infeasible and its worst score is returned.
    Otherwise the episode-score column enters the old reduction: 0.50 times
    the mean, 0.30 times the mean of the worst ceil(K/4) trials, and 0.20 times
    the minimum. Return a Python float; reject nonfinite or malformed input.
    """
    metrics = np.asarray(metrics, dtype=float)
    if metrics.ndim != 2 or not len(metrics) or metrics.shape[1] != 9:
        raise ValueError("Expected a nonempty matrix of nine episode metrics")
    if not np.isfinite(metrics).all():
        raise ValueError("Episode metrics must be finite")
    scores = np.sort(metrics[:, 0])
    if scores[0] < -1.:
        return float(scores[0])
    tail = max(1, int(np.ceil(len(scores) / 4)))
    return float(.5 * scores.mean() + .3 * scores[:tail].mean() + .2 * scores[0])


@njit(cache=True, parallel=True)
def _population_metrics(physical, currents, cases, mechanics, duration, fitness_window_s, minimum_interval=0):
    """Parallelize independent individual/episode pairs; preserve their order."""
    result = np.empty((len(physical), len(cases), 9))
    for job in prange(len(physical)*len(cases)):
        individual, episode = job//len(cases), job % len(cases)
        metrics, _ = _integrate(physical[individual], cases[episode],
                                currents[individual, episode], mechanics,
                                PARAMETERS, duration, 0, fitness_window_s, minimum_interval)
        result[individual, episode] = metrics
    return result


def evaluate_population(genome, population, cases, duration_s):
    """Decode each genome once, evaluate all cases, then reduce episode fitness.

    The bank is supplied explicitly. The GA receives only training scores;
    validation/test banks are used exclusively by their respective callers.
    Numba threads evaluate independent jobs without any random draws.
    """
    population = np.asarray(population, dtype=float)
    cases = np.asarray(cases, dtype=float)
    if population.ndim != 2 or not len(population) or population.shape[1] != genome.size:
        raise ValueError('Population must be N by the current number of independent genes')
    if cases.ndim != 2 or not len(cases) or cases.shape[1] != 5 or not np.isfinite(cases).all():
        raise ValueError('Expected a nonempty bank with five finite values per episode')
    if np.any(np.abs(cases[:, :2]) >= ANGLE_LIMIT):
        raise ValueError('Targets and initial angles must be strictly inside +/-90 degrees')
    validate_duration(duration_s)
    fitness_window_s = validate_fitness_window(
        genome.controller.data['simulation'].get('fitness_window_s', DEFAULT_FITNESS_WINDOW_S))
    decoded = [genome.decode(unit) for unit in population]
    physical = np.asarray([p for p, _ in decoded])
    currents = np.asarray([genome.currents(command, cases) for _, command in decoded])
    metrics = _population_metrics(physical, currents, cases, genome.controller.mechanics,
                                   duration_s, fitness_window_s,
                                   firing_interval_steps(genome.controller.data['simulation'].get('max_firing_rate_hz')))
    return np.array([fitness(row) for row in metrics]), metrics


def replay_genome(genome, unit, *, target_deg=0., angle_deg=-30., speed_rad_s=0.,
                  duration_s=10., stride=10, fitness_window_s=None):
    """Run one fixed-target episode for a saved genome, returning SI-unit arrays."""
    if genome.kind == 'none' and target_deg != 0:
        raise ValueError('A/B hold the upper vertical: target_deg must be zero')
    case = np.array([np.deg2rad(target_deg), np.deg2rad(angle_deg), speed_rad_s, -1., 0.])
    physical, command = genome.decode(unit)
    currents = genome.currents(command, case[None])[0]
    if fitness_window_s is None:
        fitness_window_s = genome.controller.data['simulation'].get('fitness_window_s', DEFAULT_FITNESS_WINDOW_S)
    fitness_window_s = validate_fitness_window(fitness_window_s)
    metrics, trace = simulate(physical, case, genome.controller.mechanics,
                              currents=currents, duration_s=duration_s, stride=stride,
                              fitness_window_s=fitness_window_s,
                              max_firing_rate_hz=genome.controller.data['simulation'].get('max_firing_rate_hz'))
    result = dict(metrics=metrics, trace=trace, case=case, currents=currents,
                duration_s=duration_s, fitness_window_s=fitness_window_s,
                metric_columns=np.array(METRIC_COLUMNS), trace_columns=np.array(TRACE_COLUMNS))
    if genome.controller.data['simulation'].get('max_firing_rate_hz') is not None:
        result['max_firing_rate_hz'] = genome.controller.data['simulation']['max_firing_rate_hz']
    return result


def measure_activity(result, bin_s=1.):
    """Measure both spike outputs without changing dynamics or fitness.

    Requires every 0.1 ms sample. Late statistics are available only for a
    complete episode and a fully observed fitness window. Coactivity means
    at least one spike from EACH neuron in a common bin, not coincident spikes.
    Coactivity remains diagnostic. ISIs also independently audit the optional
    rate constraint used in training; test diagnostics never select a winner.
    """
    trace = np.asarray(result['trace'])
    if trace.ndim != 2 or trace.shape[1] != len(TRACE_COLUMNS) or len(trace) < 2:
        raise ValueError('Activity requires a full-resolution nonempty replay trace')
    if not np.allclose(np.diff(trace[:, 0]), DT_MS/1000, atol=1e-10, rtol=0):
        raise ValueError('Replay with stride=1 to measure spikes; decimated traces lose events')
    duration, window = float(result['duration_s']), float(result['fitness_window_s'])
    if not np.isfinite(bin_s) or bin_s <= 0:
        raise ValueError('Activity bin width must be positive and finite')
    spikes = trace[:, 19:21] > 0
    counts = spikes.sum(axis=0)
    observed = float(trace[-1, 0])
    complete = bool(observed >= duration-1e-9 and result['metrics'][1] >= duration-1e-9)
    minimum_interval = firing_interval_steps(result.get('max_firing_rate_hz'))
    minimum_isi_ms, peak_rate_hz, violations = [], [], []
    for neuron in range(2):
        intervals = np.diff(np.flatnonzero(spikes[:, neuron]))
        minimum_isi_ms.append(float(intervals.min()*DT_MS) if len(intervals) else None)
        peak_rate_hz.append(float(1000/(intervals.min()*DT_MS)) if len(intervals) else None)
        violations.append(int((intervals < minimum_interval).sum()) if minimum_interval else 0)
    row = dict(observed_duration_s=observed, completed=complete,
               spike_count=counts.tolist(), rate_hz=(counts/observed).tolist(),
               minimum_isi_ms=minimum_isi_ms, peak_instantaneous_rate_hz=peak_rate_hz,
               max_firing_rate_hz=result.get('max_firing_rate_hz'),
               rate_limit_violations=violations,
               rate_limit_observed_pass=not any(violations) if minimum_interval else None,
               late_window_s=window, bin_s=float(bin_s), late_spike_count=None,
               late_rate_hz=None, late_bin_spike_counts=None, both_active_late=None,
               both_active_every_late_bin=None, coactive_bin_fraction=None,
               late_mean_force_N=None)
    if complete and duration >= window:
        start = duration-window
        mask = (trace[:, 0] > start+1e-12) & (trace[:, 0] <= duration+1e-12)
        late = spikes[mask].sum(axis=0)
        bins = []
        for left in np.arange(start, duration, bin_s):
            right = min(left+bin_s, duration)
            selected = (trace[:, 0] > left+1e-12) & (trace[:, 0] <= right+1e-12)
            bins.append(spikes[selected].sum(axis=0))
        bins = np.asarray(bins)
        coactive = (bins > 0).all(axis=1)
        row.update(late_spike_count=late.tolist(), late_rate_hz=(late/window).tolist(),
                   late_bin_spike_counts=bins.tolist(), both_active_late=bool((late>0).all()),
                   both_active_every_late_bin=bool(coactive.all()),
                   coactive_bin_fraction=float(coactive.mean()),
                   late_mean_force_N=trace[mask, 3:5].mean(axis=0).tolist())
    return row


def plot_replay(result):
    """Return a six-panel Matplotlib figure; importing the model needs no GUI."""
    import matplotlib.pyplot as plt
    trace = result['trace']
    if not len(trace):
        raise ValueError('A replay plot requires a positive recording stride')
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), layout='constrained')
    t = trace[:, 0]
    axes[0, 0].plot(t, np.degrees(trace[:, 1]), label='Limb')
    axes[0, 0].axhline(np.degrees(result['case'][0]), color='k', ls='--', label='Target')
    axes[0, 0].set(ylabel='Angle (deg)', xlabel='Time (s)')
    axes[0, 1].plot(np.degrees(trace[:, 1]), trace[:, 2])
    axes[0, 1].set(xlabel='Angle (deg)', ylabel='Speed (rad/s)')
    for i, side in enumerate(('Flexor', 'Extensor')):
        axes[1, 0].plot(t, trace[:, 5+i], lw=.6, label=side)
        axes[1, 1].plot(t, trace[:, 3+i], label=side)
        axes[2, 0].plot(t, trace[:, 17+i], label=side)
    axes[1, 0].set(xlabel='Time (s)', ylabel='Membrane potential (mV)')
    axes[1, 1].set(xlabel='Time (s)', ylabel='Muscle force (N)')
    axes[2, 0].set(xlabel='Time (s)', ylabel='Muscle activation')
    for i, label in enumerate(('Ia F', 'II F', 'Ib F', 'Ia E', 'II E', 'Ib E')):
        axes[2, 1].plot(t, trace[:, 11+i], lw=.8, label=label)
    axes[2, 1].set(xlabel='Time (s)', ylabel='Afferent response (model units)')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    return fig

