"""Check muscle moments independently from both implementations via tensile geometry."""
from pathlib import Path

import numpy as np
import pytest

from SpikingNN.core.Izh_net import OneDOFLimb
from scripts.AfferentBalance2026.experiments.core.genome import GenomeConfig
from scripts.AfferentBalance2026.experiments.simulation import reference_trace, simulate


def tensile_moments(theta, forces, a1, a2):
    """Project Cartesian tendon-pull forces onto displacement per unit clockwise angle."""
    attachment = a2 * np.array([np.sin(theta), np.cos(theta)])
    anchors = np.array([[a1, 0.], [-a1, 0.]])
    tendons = anchors - attachment
    lengths = np.linalg.norm(tendons, axis=1)
    forces_xy = np.asarray(forces)[:, None] * tendons / lengths[:, None]
    displacement = a2 * np.array([np.cos(theta), -np.sin(theta)])
    return forces_xy @ displacement


@pytest.mark.parametrize('angle_deg', [-50., -30., -10., 0., 10., 30., 50.])
@pytest.mark.parametrize('forces', [(10., 0.), (0., 10.), (8., 3.)])
def test_library_moment_and_acceleration_follow_tendon_pull(angle_deg, forces):
    """One active muscle must accelerate toward its anchor; external torque keeps its sign."""
    theta = np.deg2rad(angle_deg)
    limb = OneDOFLimb(m=.3, ls=.3, b=0., a1=.2, a2=.05,
                     q0=theta-np.pi/2, w0=0.)
    limb.g = 0.
    external_moment = .017
    expected = tensile_moments(theta, forces, limb.a1, limb.a2).sum() + external_moment
    limb.step(dt=.1, F_flex=forces[0], F_ext=forces[1], M=external_moment)
    np.testing.assert_allclose(limb.M_tot, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(limb.w, .0001 * expected / limb.J, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('angle_deg', [-50., 0., 50.])
@pytest.mark.parametrize('velocity', [-.7, 0., .7])
def test_tendon_work_and_lengthening_velocity_have_consistent_signs(angle_deg, velocity):
    """Tendon power equals minus tension times lengthening speed for each muscle."""
    theta = np.deg2rad(angle_deg)
    limb = OneDOFLimb(a1=.2, a2=.05, q0=theta-np.pi/2, w0=velocity)
    eps = 1e-6
    length_plus = np.array([limb.L(limb.q+eps), limb.L(np.pi-limb.q-eps)])
    length_minus = np.array([limb.L(limb.q-eps), limb.L(np.pi-limb.q+eps)])
    derivative = (length_plus-length_minus)/(2*eps)
    lengthening = np.array([limb.h(limb.L(limb.q), limb.q),
                           -limb.h(limb.L(np.pi-limb.q), np.pi-limb.q)]) * velocity
    np.testing.assert_allclose(lengthening, derivative*velocity, rtol=1e-8, atol=1e-10)
    forces = np.array([8., 3.])
    moments = tensile_moments(theta, forces, limb.a1, limb.a2)
    np.testing.assert_allclose(moments*velocity, -forces*lengthening, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('angle_deg,velocity', [(-50., .7), (0., 0.), (50., -.7)])
def test_numba_moment_and_acceleration_match_cartesian_tension(angle_deg, velocity):
    """Verify the accelerated trace against force geometry, then against library objects."""
    cfg = GenomeConfig(Path(__file__).resolve().parents[1]/'experiments/A_balance/genome_cfg.json')
    p = cfg.initial_physical
    case = np.array([0., np.deg2rad(angle_deg), velocity, -1., 0.])
    currents = np.array([150., 110.])
    _, trace = simulate(p, case, cfg.mechanics, currents=currents, duration_s=.08, stride=1)
    previous, current = trace[:-1], trace[1:]
    expected = np.array([tensile_moments(old[1], new[3:5], p[46], p[47]).sum()
                         for old, new in zip(previous, current)])
    assert np.max(current[:, 3:5]) > 0., 'The test must exercise nonzero muscle force'
    np.testing.assert_allclose(current[:, 25], expected, rtol=1e-10, atol=1e-12)
    mass, length, viscosity, gravity = cfg.mechanics
    inertia = mass*length**2/3
    acceleration = (.5*mass*gravity*length*np.sin(previous[:, 1])
                    - viscosity*previous[:, 2]+expected)/inertia
    np.testing.assert_allclose(current[:, 2]-previous[:, 2], .0001*acceleration,
                               rtol=1e-9, atol=1e-12)
    library = reference_trace(p, case, cfg.mechanics, currents=currents, duration_s=.08)
    np.testing.assert_allclose(trace, library, rtol=1e-8, atol=1e-8)
