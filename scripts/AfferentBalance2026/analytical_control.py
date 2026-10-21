"""Check analytical control estimates against the saved unscaled model.

This is an algebraic audit, not another training run or a proof of stability
of the complete spiking network. It also checks the discrete energy identity
on one previously saved A1 trajectory. Run this file with ``--help``.
"""
import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
sys.path.insert(0, str(BASE))
from report.article_figure_data import read_json, scenarios, sha256, write_json
from runtime_support import (
    activate_runtime, core_module, current_results, resolve_recorded_path,
    runtime_directory,
)


def geometry(theta, mechanics):
    """Return muscle lengths and positive moment arms inside the upright half-plane.

    ``theta`` is the angle from upright in radians. The implemented torque is
    ``-arm_f * force_f + arm_e * force_e``; no multiplier is applied.
    """
    product = mechanics.a1_m * mechanics.a2_m
    squared_length = mechanics.a1_m**2 + mechanics.a2_m**2
    length_f = np.sqrt(squared_length - 2 * product * np.sin(theta))
    length_e = np.sqrt(squared_length + 2 * product * np.sin(theta))
    return (length_f, length_e, product * np.cos(theta) / length_f,
            product * np.cos(theta) / length_e)


def mechanical_estimates(mechanics, dt_ms):
    """Compute local ranks, instability, coactivation threshold and energy costs."""
    inertia = mechanics.mass_kg * mechanics.length_m**2 / 3
    gravity = mechanics.mass_kg * mechanics.gravity_m_s2 * mechanics.length_m / 2
    length_f, _, gamma, _ = geometry(0., mechanics)
    stiffness_per_force = 2 * (mechanics.a1_m * mechanics.a2_m)**2 / length_f**3
    critical_force = gravity / stiffness_per_force
    matrix_a = np.array([[0., 1.], [gravity / inertia, -mechanics.viscosity / inertia]])
    matrix_b = np.array([[0.], [gamma / inertia]])
    controllability = np.column_stack((matrix_b, matrix_a @ matrix_b))
    step = dt_ms / 1000
    matrix_ad, matrix_bd = np.eye(2) + step * matrix_a, step * matrix_b
    discrete = np.column_stack((matrix_bd, matrix_ad @ matrix_bd))
    determinant = -(gamma / inertia)**2
    np.testing.assert_allclose(np.linalg.det(controllability), determinant)
    np.testing.assert_allclose(np.linalg.det(discrete), step**3 * determinant)
    assert np.linalg.matrix_rank(controllability) == np.linalg.matrix_rank(discrete) == 2
    # Compare the derivative derived on paper with the exact nonlinear torque.
    epsilon = 1e-5
    _, _, rf_plus, re_plus = geometry(epsilon, mechanics)
    _, _, rf_minus, re_minus = geometry(-epsilon, mechanics)
    slope = ((-rf_plus + re_plus) - (-rf_minus + re_minus)) / (2 * epsilon)
    np.testing.assert_allclose(slope, -stiffness_per_force, rtol=1e-8)
    eigenvalues = np.linalg.eigvals(matrix_a)
    np.testing.assert_allclose(eigenvalues.imag, 0.)
    eigenvalues = np.sort(eigenvalues.real)
    angles = np.array([22.5, 45., 50.])
    angle_rad = np.deg2rad(angles)
    _, _, restoring_arm, _ = geometry(angle_rad, mechanics)
    return dict(
        J_kg_m2=inertia, Kg_Nm=gravity, L0_m=float(length_f), gamma_m=float(gamma),
        A=matrix_a.tolist(), B_force=matrix_b.tolist(),
        controllability_rank=2, controllability_determinant=determinant,
        discrete_controllability_rank=2,
        discrete_controllability_determinant=float(np.linalg.det(discrete)),
        eigenvalues_per_s=eigenvalues.tolist(), growth_time_s=1 / eigenvalues[-1],
        coactivation_stiffness_per_N=float(stiffness_per_force),
        critical_equal_force_N=float(critical_force),
        energy_to_upright_J=dict(zip(map(str, angles), gravity * (1 - np.cos(angle_rad)))),
        minimum_restoring_force_N=dict(zip(
            map(str, angles), gravity * np.sin(angle_rad) / restoring_arm)),
        gravity_linearization_relative_error=dict(zip(
            map(str, angles), angle_rad / np.sin(angle_rad) - 1)),
    )


def force_estimates(controller, mechanics, critical_force):
    """Bound forces and static torques, without assuming the network can realize them."""
    parameters = np.asarray(controller, dtype=float)
    bounds = np.array([parameters[k + 1] * parameters[k + 2] * parameters[k + 4]
                       for k in (16, 23)])
    thresholds = parameters[44:46]
    times = parameters[[19, 20, 26, 27]]
    assert np.all(times >= .1)
    rows = []
    gravity = mechanics.mass_kg * mechanics.gravity_m_s2 * mechanics.length_m / 2
    for angle in (-50., -45., -22.5, 0., 22.5, 45., 50.):
        theta = np.deg2rad(angle)
        _, _, rf, re = geometry(theta, mechanics)
        required = -gravity * np.sin(theta)
        lower, upper = -rf * bounds[0], re * bounds[1]
        rows.append(dict(angle_deg=angle, required_torque_Nm=float(required),
                         lower_Nm=float(lower), upper_Nm=float(upper),
                         interior=bool(lower < required < upper),
                         margin_Nm=float(min(required - lower, upper - required))))
    return dict(force_bounds_N=bounds.tolist(), force_thresholds_N=thresholds.tolist(),
                Ib_identically_zero=(bounds <= thresholds).tolist(),
                equal_force_can_exceed_critical=bool(bounds.min() > critical_force),
                max_step_over_muscle_time=float(.1 / times.min()), torque_intervals=rows)


def check_sensor_inverse(mechanics, afferents_class):
    """Verify full-Afferents inverse formulas when activation and active lengths are known.

    Artificial threshold choices keep both length channels active in this check.
    This verifies a conditional algebraic identity, not observability of a genome.
    """
    theta, omega = np.meshgrid(np.deg2rad(np.linspace(-50, 50, 11)),
                               np.linspace(-1, 1, 13))
    lf, le, rf, re = geometry(theta, mechanics)
    lengths, velocities = (lf, le), (-rf * omega, re * omega)
    recovered_lengths, recovered_speeds = [], []
    for length, velocity, activation in zip(lengths, velocities, (.2, .7)):
        receptor = afferents_class()
        receptor.L_th = .14
        ia = receptor.Ia(velocity, length, activation)
        ii = receptor.II(length, activation)
        stretch = (ii - receptor.k_nII * activation - receptor.const_II) / receptor.k_dII
        assert np.all(stretch > 0)
        speed_term = ia - receptor.k_dI * stretch - receptor.k_nI * activation - receptor.const_I
        speed_positive = receptor.L_th * np.maximum(speed_term / receptor.k_v, 0)**(1 / receptor.p_v)
        recovered_lengths.append(receptor.L_th * (1 + stretch))
        recovered_speeds.append(speed_positive)
    reconstructed_theta = np.arcsin(
        (recovered_lengths[1]**2 - recovered_lengths[0]**2) /
        (4 * mechanics.a1_m * mechanics.a2_m))
    reconstructed_omega = (le * recovered_speeds[1] - lf * recovered_speeds[0]) / (
        mechanics.a1_m * mechanics.a2_m * np.cos(theta))
    np.testing.assert_allclose(reconstructed_theta, theta, atol=2e-14)
    np.testing.assert_allclose(reconstructed_omega, omega, atol=2e-14)
    return dict(cases=theta.size,
                max_angle_error_rad=float(np.max(abs(reconstructed_theta - theta))),
                max_velocity_error_rad_s=float(np.max(abs(reconstructed_omega - omega))))


def check_saved_energy(trace_root, mechanics, estimates):
    """Verify explicit-Euler kinematics and energy on the prespecified A1 example."""
    relative = 'balance_A1/zero_velocity/case_002.npz'
    record = next(row for row in read_json(trace_root / 'traces.json') if row['file'] == relative)
    path = trace_root / relative
    assert sha256(path) == record['sha256']
    with np.load(path, allow_pickle=False) as saved:
        trace = saved['trace']
    step = .0001
    theta, omega = trace[:, 1], trace[:, 2]
    _, _, rf, re = geometry(theta[:-1], mechanics)
    # The real scheme uses updated forces with the preceding angle and speed.
    torque = -rf * trace[1:, 3] + re * trace[1:, 4]
    inertia, gravity = estimates['J_kg_m2'], estimates['Kg_Nm']
    acceleration = (gravity * np.sin(theta[:-1]) -
                    mechanics.viscosity * omega[:-1] + torque) / inertia
    np.testing.assert_allclose(np.diff(theta), step * omega[:-1], atol=3e-15)
    np.testing.assert_allclose(np.diff(omega), step * acceleration, atol=3e-14)
    energy = .5 * inertia * omega**2 + gravity * np.cos(theta)
    remainder = .5 * inertia * step**2 * acceleration**2 + gravity * (
        np.cos(theta[:-1] + step * omega[:-1]) - np.cos(theta[:-1]) +
        step * omega[:-1] * np.sin(theta[:-1]))
    predicted = step * (torque * omega[:-1] - mechanics.viscosity * omega[:-1]**2) + remainder
    error = float(np.max(abs(np.diff(energy) - predicted)))
    assert error < 3e-15
    late_forces = trace[trace[:, 0] >= 8, 3:5]
    means = late_forces.mean(axis=0)
    return dict(file=str(path.relative_to(REPO)), sha256=record['sha256'],
                checked_steps=len(trace) - 1, max_energy_identity_error_J=error,
                mean_force_8_to_10_s_N=means.tolist(),
                min_force_8_to_10_s_N=late_forces.min(axis=0).tolist(),
                max_force_8_to_10_s_N=late_forces.max(axis=0).tolist(),
                mean_force_stiffness_Nm_per_rad=float(
                    estimates['coactivation_stiffness_per_N'] * means.mean()))


def main():
    """Write reproducible estimates and checks while leaving all simulation data intact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=current_results())
    parser.add_argument('--trace-root', type=Path,
                        default=BASE / 'report/results/article_phase_grid_20261019')
    parser.add_argument('--output', type=Path, default=BASE / 'report/analytical/estimates.json')
    args = parser.parse_args()
    root, trace_root = args.root.resolve(), args.trace_root.resolve()
    protocol_path = trace_root / 'protocol.json'
    protocol = read_json(protocol_path)
    assert resolve_recorded_path(protocol['source_study']) == root
    runtime = runtime_directory(root)
    activate_runtime(runtime)
    config = core_module('config')
    mechanics = config.MECHANICS
    assert mechanics.muscle_torque_factor == 1 and config.DT_MS == .1
    assert protocol['mechanics'] == asdict(mechanics)
    assert sha256(root / 'factor_1/runtime_provenance.json') == protocol['runtime_provenance_sha256']
    selected = scenarios(root, runtime)  # Verifies saved generation and numerical-source hashes.
    estimates = mechanical_estimates(mechanics, config.DT_MS)
    bounds = {}
    for scenario in selected[:4]:
        bounds[scenario['name']] = force_estimates(
            scenario['controller'], mechanics, estimates['critical_equal_force_N'])
        original = next(row for row in protocol['scenarios'] if row['name'] == scenario['name'])
        np.testing.assert_array_equal(scenario['controller'], original['controller'])
    library = importlib.import_module('SpikingNN.core.Izh_net')
    checks = dict(sensor_inverse=check_sensor_inverse(mechanics, library.Afferents),
                  saved_A1=check_saved_energy(trace_root, mechanics, estimates))
    result = dict(
        status='passed', scope='Mechanical input idealization and conditional sensor identities; '
        'not a full-network stability proof.', dt_ms=.1, muscle_torque_factor=1,
        mechanics=asdict(mechanics), mechanical_estimates=estimates,
        controllers=bounds, checks=checks,
        source_study=str(root.relative_to(REPO)),
        inputs_sha256={str(protocol_path.relative_to(REPO)): sha256(protocol_path),
                       str(Path(__file__).resolve().relative_to(REPO)): sha256(__file__)},
        selected_genomes=[row['provenance'] for row in selected[:4]],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(dict(output=str(args.output), **estimates, checks=checks),
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
