"""Fixed-phase sine tracking objective and diagnostics; no command injection."""
import numpy as np
from numba import njit, prange
from dynamics import DT_S, integrate

METRIC_COLUMNS = ('fitness', 'rmse_deg', 'velocity_rmse_deg_s', 'mean_force_N',
                  'max_error_deg', 'survival_s', 'failure')


def task_vector(task):
    """Validate the target and scoring window before entering compiled code."""
    keys = ('frequency_hz', 'amplitude_deg', 'center_deg', 'phase_rad', 'duration_s',
            'score_start_s', 'velocity_weight', 'angle_limit_deg', 'tonic_drive')
    values = np.asarray([task[k] for k in keys], dtype=float)
    f, a, c, ph, duration, start, weight, limit, drive = values
    if (not np.isfinite(values).all() or f <= 0 or a <= 0 or start < 0 or start >= duration
            or duration-start < 2/f or weight < 0 or not 0 < limit < 90 or drive < 0
            or abs(c-90)+a >= limit
            or not np.isclose(duration/DT_S, round(duration/DT_S), rtol=0, atol=1e-7)):
        raise ValueError('Invalid target; score at least two cycles inside the allowed angular envelope')
    return values


def reference(time_s, task):
    """Absolute joint angle (radians) and analytic velocity; phase is fixed at t=0."""
    omega = 2*np.pi*task['frequency_hz']
    amplitude = np.deg2rad(task['amplitude_deg'])
    phase = omega*np.asarray(time_s) + task['phase_rad']
    return np.deg2rad(task['center_deg'])+amplitude*np.sin(phase), amplitude*omega*np.cos(phase)


@njit(cache=True)
def measure(trace, completed, failure, spec):
    """F=1/(1+mean((q-q*)/A)^2 + w_v*mean((v-v*)/(A*omega))^2).

    No fitted phase, lag alignment, amplitude rescaling or target feedback.
    Any numerical failure/excursion receives -1, including after the score window.
    """
    if failure or completed < round(spec[4]/DT_S):
        return np.array([-1., 1e6, 1e6, 0., 1e6, completed*DT_S, float(max(1, failure))])
    omega = 2*np.pi*spec[0]
    amplitude, center = np.deg2rad(spec[1]), np.deg2rad(spec[2])
    sq, vsq, peak, force, count = 0., 0., 0., 0., 0
    for i in range(len(trace)):
        t = trace[i, 0]
        if t < spec[5] or t >= spec[4]:
            continue
        desired = center+amplitude*np.sin(omega*t+spec[3])
        velocity = amplitude*omega*np.cos(omega*t+spec[3])
        e, ev = trace[i, 1]-desired, trace[i, 2]-velocity
        sq += e*e
        vsq += ev*ev
        peak = max(peak, abs(e))
        force += trace[i, 3]+trace[i, 4]
        count += 1
    if count == 0:
        return np.array([-1., 1e6, 1e6, 0., 1e6, completed*DT_S, 3.])
    loss = sq/count/amplitude**2 + spec[6]*vsq/count/(amplitude*omega)**2
    return np.array([1/(1+loss), np.rad2deg(np.sqrt(sq/count)),
                     np.rad2deg(np.sqrt(vsq/count)), force/count, np.rad2deg(peak),
                     completed*DT_S, 0.])


@njit(cache=False, parallel=True)
def evaluate_population(physical, spec):
    """Parallel deterministic evaluations; target never enters the physical step.

    No disk cache here: Numba does not invalidate callers when an imported
    integrate() implementation changes. Recompile this small driver once/run.
    """
    result = np.empty((len(physical), len(METRIC_COLUMNS)))
    for i in prange(len(physical)):
        trace, completed, failure = integrate(physical[i], round(spec[4]/DT_S), spec[8],
                                             10, False, np.deg2rad(spec[7]))
        result[i] = measure(trace, completed, failure, spec)
    return result


def dominant_frequency(time_s, signal, minimum=.1, maximum=15.):
    """Dominant non-DC spectral peak, Hann window and parabolic interpolation.

    This is a diagnostic, not an independently guaranteed fundamental frequency.
    Finite-window resolution remains approximately 1/window_duration.
    """
    t, y = np.asarray(time_s), np.asarray(signal)
    if len(t) < 8 or np.ptp(y) < 1e-8:
        return None
    nfft = 2**int(np.ceil(np.log2(len(y)*4)))
    power = abs(np.fft.rfft((y-y.mean())*np.hanning(len(y)), n=nfft))**2
    freq = np.fft.rfftfreq(nfft, float(t[1]-t[0]))
    indices = np.flatnonzero((freq >= minimum) & (freq <= maximum))
    if not len(indices):
        return None
    k = indices[np.argmax(power[indices])]
    delta = 0.
    if 0 < k < len(power)-1:
        a, b, c = np.log(power[k-1:k+2]+1e-300)
        if a-2*b+c != 0:
            delta = np.clip(.5*(a-c)/(a-2*b+c), -.5, .5)
    return float((k+delta)*(freq[1]-freq[0]))


def diagnostics(trace, task, *, start_s=None, end_s=None):
    """Report bias, amplitude, frequency and fixed-reference phase mismatch."""
    start = task['score_start_s'] if start_s is None else start_s
    end = task['duration_s'] if end_s is None else end_s
    block = trace[(trace[:, 0] >= start) & (trace[:, 0] < end)]
    if len(block) < 8:
        raise ValueError('Not enough samples for diagnostics')
    t, q = block[:, 0], np.rad2deg(block[:, 1])
    theta = 2*np.pi*task['frequency_hz']*t
    design = np.column_stack((np.ones(len(t)), np.sin(theta), np.cos(theta)))
    dc, s, c = np.linalg.lstsq(design, q, rcond=None)[0]
    amplitude = np.hypot(s, c)
    phase_error = np.angle(np.exp(1j*(np.arctan2(c, s)-task['phase_rad'])))
    target, _ = reference(t, task)
    result = dict(window_s=[float(start), float(end)], rmse_deg=float(np.sqrt(np.mean((q-np.rad2deg(target))**2))),
                  mean_angle_deg=float(q.mean()), robust_half_range_deg=float(np.ptp(np.percentile(q, [5, 95]))/2),
                  fitted_target_amplitude_deg=float(amplitude), phase_error_deg=float(np.rad2deg(phase_error)),
                  dominant_motion_frequency_hz=dominant_frequency(t, q),
                  harmonic_residual_rmse_deg=float(np.sqrt(np.mean((q-design@np.array([dc,s,c]))**2))))
    if trace.shape[1] >= 35:
        # Population burst frequency from 50-ms spike-count bins, avoiding spike-rate aliases.
        width = max(1, round(.05/(t[1]-t[0])))
        indices = np.arange(0, len(block), width)
        if len(indices) > 8:
            rates = np.diff(block[indices, 31:33], axis=0)/.05
            bt = t[indices][1:]
            result['RG_F_burst_frequency_hz'] = dominant_frequency(bt, rates[:, 0], maximum=8.)
            result['RG_E_burst_frequency_hz'] = dominant_frequency(bt, rates[:, 1], maximum=8.)
    return result
