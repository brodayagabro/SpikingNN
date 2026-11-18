"""Continuous reference signals and a streaming Numba trajectory integrator.

Commands depend only on the requested angle. Physical state feedback is provided
exclusively by the full nonlinear afferents in the shared physical step.
"""
import numpy as np
from numba import njit

from ..simulation import (ANGLE_LIMIT, PARAMETERS, TARGET, initial_state,
                          physical_step, write_state)

DT_S = .0001
METRIC_COLUMNS = ('score', 'survival_s', 'tracking_rmse_rad', 'peak_error_rad',
                  'mean_force_N', 'success', 'in_band_fraction',
                  'velocity_rmse_rad_s', 'final_error_rad')


def step_count(duration_s):
    """Validate a positive duration on the fixed 0.1 ms integration grid."""
    if (not np.isfinite(duration_s) or duration_s < DT_S
            or not np.isclose(duration_s / DT_S, round(duration_s / DT_S),
                              rtol=0, atol=1e-7)):
        raise ValueError('Duration must be positive and a multiple of 0.0001 s')
    return round(duration_s / DT_S)


def sine_reference(spec):
    """Return desired angle and its analytic velocity, including the reset sample.

    A constant warm-up is followed by a quintic onset and two complete cycles.
    The onset interpolates between the initial sinusoid value and the sinusoid;
    it never extends the requested offset +/- amplitude envelope.
    """
    amplitude, offset, phase = (spec[k] for k in ('amplitude_deg', 'offset_deg', 'phase_rad'))
    period, hold, ramp = (spec[k] for k in ('period_s', 'hold_s', 'ramp_s'))
    if (not np.isfinite(list(spec.values())).all() or amplitude <= 0
            or abs(offset) + amplitude > 45 or period <= 0 or hold < 0 or ramp <= 0):
        raise ValueError('Invalid sine envelope or timing')
    steps = step_count(spec['duration_s'])
    t = np.arange(steps+1) * DT_S
    z = np.maximum(t-hold, 0.)
    u = np.clip(z/ramp, 0., 1.)
    envelope = 6*u**5 - 15*u**4 + 10*u**3
    derivative = 30*u**2*(u-1)**2/ramp
    w = 2*np.pi/period
    delta = np.sin(w*z+phase) - np.sin(phase)
    angle = offset + amplitude*(np.sin(phase) + envelope*delta)
    velocity = amplitude*(derivative*delta + envelope*w*np.cos(w*z+phase))
    return np.ascontiguousarray(np.deg2rad(np.column_stack((angle, velocity))))


def sine_case(amplitude=30., period=20., offset=0., phase=0., initial_angle=0.,
              initial_speed=0., hold=4., ramp=2.):
    """Specify one continuous episode with at least two cycles after the onset."""
    duration = np.ceil((hold+ramp+2*period)/DT_S-1e-7)*DT_S
    return dict(amplitude_deg=float(amplitude), period_s=float(period),
                offset_deg=float(offset), phase_rad=float(phase),
                initial_angle_deg=float(initial_angle), initial_speed_rad_s=float(initial_speed),
                hold_s=float(hold), ramp_s=float(ramp), duration_s=float(duration))


def trajectory_banks(settings):
    """Create disjoint seeded banks; test trajectories never select a genome."""
    seeds = settings['bank_seeds']
    if len(set(seeds.values())) != 3:
        raise ValueError('Use three independent seeds')
    banks = {}
    for name in ('training', 'validation', 'test'):
        count = settings[f'{name}_count']
        if count < 2:
            raise ValueError('Each bank must contain at least two trajectories')
        rng = np.random.default_rng(seeds[name])
        unit = np.column_stack([(rng.permutation(count)+rng.random(count))/count for _ in range(6)])
        cases = []
        for a, f, o, phase, theta, omega in unit:
            amplitude = 10+35*a
            frequency = .04+.06*f
            cases.append(sine_case(amplitude, 1/frequency, (2*o-1)*(45-amplitude),
                                   2*np.pi*phase, -45+90*theta, -.5+omega,
                                   settings['hold_s'], settings['ramp_s']))
        if name == 'training':
            cases[0] = sine_case(45, 20, initial_angle=-30,
                                 hold=settings['hold_s'], ramp=settings['ramp_s'])
            cases[1] = sine_case(30, 10, initial_angle=30, initial_speed=.3,
                                 hold=settings['hold_s'], ramp=settings['ramp_s'])
        banks[name] = cases
    return banks


@njit(cache=True)
def command_currents(coefficients, target_rad):
    """Evaluate ascending-power polynomial coefficients at target_rad / pi/4."""
    x = target_rad / (np.pi/4)
    currents = np.zeros(2)
    for i in range(2):
        # Ascending powers match the established C command calculation exactly.
        for k in range(coefficients.shape[1]):
            currents[i] += coefficients[i, k]*x**k
    return currents


@njit(cache=True)
def stream(p, coefficients, mechanics, state, current, reference, tau_s,
           scoring_start, stride, start_step=0):
    """Advance state/current in place, recording optional post-step observations.

    Reference row n commands interval [n*dt, (n+1)*dt); row n+1 scores its
    end state. Recorded currents are those actually applied during that interval.
    Missing samples after a fall receive zero reward, not an extrapolated reward.
    """
    steps = len(reference)-1
    trace = np.zeros(((steps+max(stride, 1)-1)//max(stride, 1)+2 if stride else 0, 26))
    commands = np.zeros((len(trace), 4))
    indices = np.zeros(len(trace), dtype=np.int64)
    row = 1 if stride else 0
    if stride:
        write_state(trace[0], state, start_step*DT_S)
        commands[0, :2] = current
        commands[0, 2:] = command_currents(coefficients, reference[0, 0])
    alpha = 1. if tau_s == 0 else -np.expm1(-DT_S/tau_s)
    quality, speed_quality, square, speed_square, peak, effort = 0., 0., 0., 0., 0., 0.
    band, count, completed, failure = 0, 0, 0, 0
    for step in range(steps):
        desired = command_currents(coefficients, reference[step, 0])
        for i in range(2):
            current[i] = desired[i] if tau_s == 0 else current[i]+alpha*(desired[i]-current[i])
        physical_step(p, mechanics, PARAMETERS, state, current)
        completed = step+1
        if not np.isfinite(state).all():
            failure = 2
            break
        error = state[0]-TARGET-reference[step+1, 0]
        velocity_error = state[1]-reference[step+1, 1]
        effort += state[14]+state[15]
        if step >= scoring_start:
            quality += np.exp(-(error/np.deg2rad(5.))**2)
            speed_quality += np.exp(-(velocity_error/.15)**2)
            square += error*error
            speed_square += velocity_error*velocity_error
            peak = max(peak, abs(error))
            band += int(abs(error) <= np.deg2rad(3.) and abs(velocity_error) <= .35)
            count += 1
        if abs(state[0]-TARGET) > ANGLE_LIMIT:
            failure = 1
        if stride and (completed % stride == 0 or completed == steps or failure):
            write_state(trace[row], state, (start_step+completed)*DT_S)
            commands[row, :2], commands[row, 2:] = current, desired
            indices[row] = completed
            row += 1
        if failure:
            break
    denominator = max(1, steps-scoring_start)
    rmse = np.sqrt(square/count) if count else ANGLE_LIMIT
    vrmse = np.sqrt(speed_square/count) if count else 10.
    fraction = band/denominator
    success = 1. if failure == 0 and completed == steps and fraction >= .95 and rmse <= np.deg2rad(3.) else 0.
    score = .1*completed/steps + (.55*quality+.2*speed_quality+.15*band)/denominator
    final_error = state[0]-TARGET-reference[completed, 0] if failure != 2 else ANGLE_LIMIT
    if failure == 2:
        score = -1.
    metrics = np.array([score, completed*DT_S, rmse, peak, effort/max(1, completed),
                        success, fraction, vrmse, final_error])
    return metrics, trace[:row], commands[:row], indices[:row], completed, failure


def replay(physical, coefficients, mechanics, spec, *, tau_s=0., stride=1, reference=None):
    """Reset once, then follow a continuous trajectory without hidden state resets."""
    if not np.isfinite(tau_s) or tau_s < 0 or not isinstance(stride, int) or stride < 0:
        raise ValueError('Use a nonnegative filter time and integer recording stride')
    ref = sine_reference(spec) if reference is None else reference
    state = initial_state(physical, np.deg2rad(spec['initial_angle_deg']), spec['initial_speed_rad_s'])
    current = command_currents(coefficients, ref[0, 0])
    metrics, trace, commands, indices, completed, failure = stream(
        physical, coefficients, mechanics, state, current, ref, tau_s,
        round(spec['hold_s']/DT_S), stride)
    return dict(metrics=metrics, trace=trace, commands=commands,
                reference=ref[indices] if stride else np.empty((0, 2)),
                completed=completed, failure=failure)


def measure_tracking_trace(trace, reference, spec):
    """Independently recompute metrics from a complete finite 0.1 ms recording."""
    steps = step_count(spec['duration_s'])
    start = round(spec['hold_s']/DT_S)
    n = len(trace)-1
    if (len(reference) != len(trace) or not np.isfinite(trace).all()
            or not np.allclose(np.diff(trace[:, 0]), DT_S, rtol=0, atol=1e-10)):
        raise ValueError('Expected a finite full-resolution trace and matching reference')
    error = trace[1:, 1]-reference[1:, 0]
    velocity = trace[1:, 2]-reference[1:, 1]
    e, v = error[start:], velocity[start:]
    rmse = np.sqrt(np.mean(e**2)) if len(e) else ANGLE_LIMIT
    vrmse = np.sqrt(np.mean(v**2)) if len(v) else 10.
    band = (abs(e) <= np.deg2rad(3.)) & (abs(v) <= .35)
    fraction = band.sum()/max(1, steps-start)
    success = float(n == steps and abs(trace[-1, 1]) <= ANGLE_LIMIT and fraction >= .95 and rmse <= np.deg2rad(3.))
    score = .1*n/steps + (.55*np.exp(-(e/np.deg2rad(5.))**2).sum()
                         + .2*np.exp(-(v/.15)**2).sum()+.15*band.sum())/max(1, steps-start)
    return np.array([score, n*DT_S, rmse, abs(e).max() if len(e) else 0.,
                     trace[1:, 3:5].sum(axis=1).mean(), success, fraction, vrmse, error[-1]])
