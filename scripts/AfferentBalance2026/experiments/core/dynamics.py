"""One Numba integrator for all protocols; independent replay uses SpikingNN classes.

Neurons/muscles use milliseconds; limb mechanics use seconds. Receptor input
is muscle activation x, as in Afferented_Limb.calc_afferents (not spike output).
"""
import numpy as np
from numba import njit
from .afferents import afferent_values
from .config import DT_MS, MECHANICS

MASS, LENGTH, VISCOSITY = MECHANICS.mass_kg, MECHANICS.length_m, MECHANICS.viscosity
A1, A2 = MECHANICS.a1_m, MECHANICS.a2_m
GRAVITY, TORQUE_FACTOR = MECHANICS.gravity_m_s2, MECHANICS.muscle_torque_factor
INERTIA = MASS * LENGTH**2 / 3
TARGET = -np.pi / 2
ANGLE_LIMIT = np.pi / 2
SETTLE_ANGLE = np.deg2rad(3.)
SETTLE_SPEED = .35
SETTLE_DEADLINE = 3.
METRIC_COLUMNS = ['score', 'survival_s', 'rmse_rad', 'peak_error_rad', 'mean_force_N',
                  'success', 'settling_s', 'late_rmse_rad', 'final_theta_rad']
TRACE_COLUMNS = ['time_s', 'theta_rad', 'omega_rad_s', 'force_flex_N', 'force_ext_N',
                 'V_flex', 'V_ext', 'U_flex', 'U_ext', 'calcium_flex', 'calcium_ext',
                 'Ia_F', 'II_F', 'Ib_F', 'Ia_E', 'II_E', 'Ib_E']

@njit(cache=True)
def rollout(p, case, target_rad=0., duration_s=10., dt_ms=DT_MS, mode=0, stride=0, angle_limit=ANGLE_LIMIT):
    """case = [theta0, omega0, kick_time_s, kick_delta_omega].

    mode: 0 neural, 1 afferents disconnected, 2 passive, 3 torque PD.
    Metrics follow METRIC_COLUMNS. Failure boundary: ±90 degrees.
    Success: settle inside ±3 degrees and |omega| <= .35 by t=3s, then stay.
    Initial states/forces are reset; optional kicks come from the case bank.
    """
    if not np.isfinite(duration_s) or duration_s < DT_MS / 1000:
        raise ValueError("duration_s must include at least one step")
    if dt_ms != DT_MS:
        raise ValueError("All experiments require dt_ms=0.1")
    if mode < 0 or mode > 3 or stride < 0:
        raise ValueError("Invalid mode or stride")
    steps = int(round(duration_s * 1000 / dt_ms))
    trace = np.empty(((steps + max(stride, 1) - 1) // max(stride, 1) + 1 if stride else 0, 17))
    row = 0
    q, omega = TARGET + case[0], case[1]
    v, u = p[4:6].copy(), p[4:6] * p[2:4]
    syn = np.zeros((2, 2))
    output = np.zeros(2)
    new_output = np.zeros(2)
    calcium, force, aff = np.zeros(2), np.zeros(2), np.zeros(6)
    activation = np.zeros(2)
    reward, error2, peak, effort = 0., 0., abs(case[0]-target_rad), 0.
    kick_step = int(round(case[2] * 1000 / dt_ms)) if case[2] >= 0 else -1
    completed = 0
    late_quality, late_error2, late_count = 0., 0., 0
    last_outside = 0.
    late_start = int(round(max(0., duration_s-2.)*1000/dt_ms))
    late_steps = max(1, steps-late_start)
    theta = case[0]
    for step in range(steps):
        if step == kick_step:
            omega += case[3]
        if mode < 2:
            for i in range(2):
                feedback = 0.
                if mode == 0:
                    for j in range(6):
                        feedback += p[30+6*i+j] * aff[j]
                current = p[46+i] + feedback
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
        torque = TORQUE_FACTOR * (force[0]*hf - force[1]*he)
        if mode == 3:
            torque = max(-1., min(1., -1.5*(q-TARGET-target_rad) - .22*omega))
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
                velocity, length, force[i], activation[i], p[42+i], p[44+i])
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



def reference_trace(p, case, duration_s=10., dt_ms=DT_MS, mode=0, target_rad=0., angle_limit=ANGLE_LIMIT):
    """Independent execution through original core classes (no JIT equations)."""
    from SpikingNN.core.Izh_net import Izhikevich_IO_Network, OneDOFLimb, SimpleAdaptedMuscle, Afferents
    if dt_ms != DT_MS or not np.isfinite(duration_s) or duration_s < DT_MS/1000:
        raise ValueError("Use dt_ms=0.1 and a positive episode duration")
    net = Izhikevich_IO_Network(N=2, input_size=2, output_size=2, afferent_size=6,
                               a=p[0:2], b=p[2:4], c=p[4:6], d=p[6:8],
                               W=p[8:12].reshape(2, 2), tau_syn=p[12:16].reshape(2, 2),
                               Q_app=np.eye(2), Q_aff=p[30:42].reshape(2, 6), P=p[48:52].reshape(2, 2))
    muscles = [SimpleAdaptedMuscle(**dict(zip(('w', 'N', 'A', 'tau_c', 'tau_1', 'm', 'k'), p[16+7*i:23+7*i])))
               for i in range(2)]
    limb = OneDOFLimb(m=MASS, ls=LENGTH, b=VISCOSITY, a1=A1, a2=A2, q0=TARGET+case[0], w0=case[1])
    receptors = [Afferents(), Afferents()]
    for i, receptor in enumerate(receptors):
        receptor.L_th, receptor.F_th = p[42+i], p[44+i]
    aff = np.zeros(6)
    history = []
    kick_step = round(case[2]*1000/dt_ms) if case[2] >= 0 else -1
    for step in range(round(duration_s*1000/dt_ms)):
        if step == kick_step:
            limb.w += case[3]
            limb.w_prev = limb.w
        if mode < 2:
            net.step(dt=dt_ms, Iapp=p[46:48], Iaff=aff if mode == 0 else np.zeros(6))
            for i, muscle in enumerate(muscles):
                muscle.step(dt=dt_ms, u=net.V_out[i])
        torque = np.clip(-1.5*(limb.q-TARGET-target_rad)-.22*limb.w, -1., 1.) if mode == 3 else 0.
        limb.step(dt=dt_ms, F_flex=muscles[0].F, F_ext=muscles[1].F, M=torque)
        for i, receptor in enumerate(receptors):
            angle = limb.q if i == 0 else np.pi-limb.q
            length = limb.L(angle)
            velocity = limb.h(length, angle)*limb.w*(1 if i == 0 else -1)
            aff[3*i:3*i+3] = [receptor.Ia(velocity, length, muscles[i].x), receptor.II(length, muscles[i].x), receptor.Ib(muscles[i].F)]
        history.append([(step+1)*dt_ms/1000, limb.q-TARGET, limb.w,
                        muscles[0].F, muscles[1].F, *net.V_prev, *net.U_prev,
                        muscles[0].Cn, muscles[1].Cn, *aff])
        if not np.isfinite(history[-1]).all() or abs(limb.q-TARGET) > angle_limit:
            break
    return np.asarray(history)


def measure_trace(trace, target_rad=0., duration_s=10., dt_ms=DT_MS, angle_limit=ANGLE_LIMIT, initial_theta=None):
    """Independent vectorized metrics for full-resolution original-core traces."""
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
