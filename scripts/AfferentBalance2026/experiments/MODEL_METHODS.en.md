# Active A/B/C mathematical model

[Русский](MODEL_METHODS.md) · [Experiments](README.en.md)

## Coordinates and mechanics

The plotted angle is $\theta=q+\pi/2$: vertical unstable equilibrium is $\theta=0$.

$$J\ddot q=\frac{mgL}{2}\cos q-\eta\dot q-F_f h(q)+F_e h(\pi-q),\qquad J=mL^2/3.$$

$$\ell(q)=\sqrt{a_1^2+a_2^2-2a_1a_2\cos q},\qquad h(q)=a_1a_2\sin q/\ell(q).$$

There is no torque multiplier. Mechanics uses seconds; neural and muscle updates use milliseconds, with the explicit conversion $dt_s=10^{-3}dt_{ms}$. The common step is 0.1 ms. Force and angle are updated in the same order as the library; muscle torque uses the pre-update angle. Positive muscle lengthening speeds are $v_f=h(q)\dot q$ and $v_e=-h(\pi-q)\dot q$.

For positive tendon tension, $M_i=-F_i\,\partial\ell_i/\partial q$. The muscle-moment sign correction is identified as `SpikingNN.Afferents/v3-tensile`. Lengthening velocities and the sign of external torque are unchanged. Earlier v2 results require re-evaluation.

## Neural and muscle states

$$\dot V_i=0.04V_i^2+5V_i+140-U_i+I_i+\sum_j S_{ij}+\sum_r Q_{ir}A_r,\quad
\dot U_i=a_i(b_iV_i-U_i).$$

At threshold $V_i\ge30$, apply the library reset $V_i\leftarrow c_i$, $U_i\leftarrow U_i+d_i$ on the next update; the spike output is 30 or 0. Recurrent currents obey $\dot S_{ij}=-S_{ij}/\tau_{ij}+W_{ij}y_j$. Muscle drive is $u_k=\sum_i P_{ki}y_i$.

$$\dot C_k=w_k u_k-C_k/\tau_{c,k},\qquad
x_k=\frac{C_k^{m_k}}{C_k^{m_k}+k_k^{m_k}},\qquad
\dot F_k=A_k N_k x_k-F_k/\tau_{1,k}.$$

Here muscle $N_k$ scales force production in the muscle equation; it is not the number of network neurons. The architecture has two neurons. Muscle activation $x$ is distinct from calcium $C$.

## Full nonlinear Afferents

Let $[z]_+=\max(z,0)$, $v_n=[v/L_{th}]_+$, $d_n=[(\ell-L_{th})/L_{th}]_+$, $f_n=[(F-F_{th})/F_{th}]_+$.

$$I_a=k_vv_n^{p_v}+k_{dI}d_n+k_{nI}x+c_I,$$
$$II=k_{dII}d_n+k_{nII}x+c_{II},\qquad I_b=k_f f_n.$$

All nine constants are read from `SpikingNN.core.Izh_net.Afferents` and recorded in run metadata. Length and force thresholds are genome fields. Six channels are ordered `Ia_F, II_F, Ib_F, Ia_E, II_E, Ib_E`. Afferents are computed after the mechanical step and affect the next neural update. No target-error term or PD controller is added. In A/B, both applied currents are exactly zero. C computes constant episode currents externally.

## Fitness and search

For target error $e=\theta-\theta_\mathrm{target}$, an episode ends after 10 s or a fall beyond ±90°. Let $n$ be completed steps, $T$ the planned step count, and $T_L$ the planned final-two-second step count. Missing steps after a fall contribute zero to normalized quality sums.

$$s=0.10\frac nT+\frac{0.30}{T}\sum_{t\le n}\max\{0,e^{-(e_t/15^\circ)^2}-0.05\min(\omega_t^2,4)\}
+\frac{0.45}{T_L}\sum_{t\in L,\ t\le n}e^{-(e_t/3^\circ)^2}+0.15\,\mathrm{success}.$$

Success requires survival and remaining within $|e|\le3^\circ$, $|\omega|\le0.35$ rad/s after a settling time no later than 3 s, with at least two final seconds inside that band. Nonfinite numerical trajectories receive score −1. Muscle effort is recorded, not penalized in this fitness.

$$\mathcal F=0.50\operatorname{mean}(s)+0.30\operatorname{mean}(\text{worst }\lceil K/4\rceil\text{ scores})+0.20\min(s).$$

The GA maximizes this same score in A/B/C. C aggregates it over all target/initial-state pairs. It searches normalized coordinates in [0,1]: 48 for both A and B, with linear or logarithmic physical mappings from JSON. Elitism retains the top ceil(5%); tournaments sample three candidates; crossover mixes parents, Gaussian mutation is reflected at the bounds, and 5% of generated children are uniform immigrants. Mutation sigma decreases from 0.20 to 0.015 on a cosine schedule. Sweeps hold per-gene mutation probability constant. Fixed `a1/a2`, plus both `A=0.37` values in A/B/C/D, are inserted after decoding, outside every genetic operation. The final population is ranked on independent validation cases; only its winner is scored on the independent test bank. Test outcomes do not guide training.

## Fixed field order

Indices are zero-based. All per-field descriptions and library parameter mappings are English in each `genome_cfg.json`.

| Indices | Fields |
|---|---|
| 0–7 | Neuron a, b, c, d, each for two neurons |
| 8–11 | Recurrent W[target, source] |
| 12–15 | Recurrent decay times, ms |
| 16–22 | Flexor w, N, A, tau_c, tau_1, m, k |
| 23–29 | Extensor w, N, A, tau_c, tau_1, m, k |
| 30–41 | Q_aff[neuron, afferent channel] |
| 42–43 | Length thresholds |
| 44–45 | Force thresholds |
| 46–47 | Fixed geometry a1, a2, metres |
| 48–51 | P[muscle, neuron] |

Mass, link length, viscosity and gravity are configuration metadata, not genes. External currents are separate simulation inputs. Schema: `afferent_controller_52_v4`.

## C: combined controller and current-law genome

For each neuron, the constant episode command is

$$I_i(\theta_*)=\sum_{k=0}^{d}c_{ik}u^k,\quad u=\theta_*/(\pi/4),\quad d\in\{2,3\}.$$

The shared 52-field block is followed by coefficients in neuron-major, ascending-power order. C fixes `a1`, `a2` and both muscle `A=0.37` fields. A cubic therefore has 60 physical fields and 56 mutable coordinates; a quadratic has 58/54. Combined schema: `afferent_setpoint_joint_v1`. Polynomial coefficients are bounded at ±1000 nA; the polynomial output is not clipped and may exceed this value.

Every candidate shares one controller across five training targets. A staged option replaces polynomial genes temporarily with ten target-specific currents, fits bounded least-squares polynomials in u, then jointly refines controller and coefficients. The fit approximates currents, so its dynamical performance is re-evaluated. Training uses 5×8 cases, polynomial validation 9×12 and test 17×24, with disjoint initial states. Eight test-only angles check generalization beyond training and validation targets. Each episode starts with reset internal states; this protocol does not test online switching between target commands. Total training generations are matched across methods; validation overhead can differ. No measured-state signal is added to the command law.

## D extension: a moving reference

[D_trajectory](D_trajectory/README.en.md) uses the same physics continuously: each step maps the desired angle into two polynomial currents, without resetting physical state. `AngleController.step/advance` provides the streaming interface. Desired angular velocity is used only for evaluation, never supplied to neurons. The static settling criterion is replaced by the explicit tracking criterion documented in D; robust aggregation remains shared.


The 26 October corrected-mechanics retraining selects across validation checkpoints, including generation zero. Its full scientific description, actual commands and results are in [the current article](../report/REPORT.en.md). Historical sweep protocols above remain distinguishable from the new warm-start series. Applied-current units are model units; legacy nA key names do not establish physical calibration.
