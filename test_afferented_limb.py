"""Проверки внешних афферентов. Tests for additive limb feedback.

Run from the repository root: python -m pytest test_afferented_limb.py -v.
The installed package uses SpikingNN.core.Izh_net, not a top-level Izh_net.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Type

import numpy as np
import pytest

from SpikingNN.core.Izh_net import (
    Afferented_Limb,
    Izhikevich_IO_Network,
    Net_Limb_connect,
    OneDOFLimb,
    SimpleAdaptedMuscle,
)
from SpikingNN.core.Var_Limb import Var_Limb


@pytest.fixture(
    params=[False, True], ids=["full-afferents", "simple-afferents"],
)
def active_limb(request: pytest.FixtureRequest) -> Afferented_Limb:
    """Создать активную конечность. Build a limb with nonzero intrinsic input.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Selects the full or simplified afferent model.

    Returns
    -------
    Afferented_Limb
        Independently initialized limb after one driven step.
    """
    limb = Afferented_Limb(
        Limb=OneDOFLimb(q0=1.2, w0=0.2),
        use_simple_afferents=request.param,
    )
    limb.step(dt=0.1, uf=0.5, ue=0.3)
    return limb


def _state(limb: Afferented_Limb) -> np.ndarray:
    """Снять состояние. Snapshot mechanical, muscle and afferent state.

    Parameters
    ----------
    limb : Afferented_Limb
        Object whose mutable state is being checked.

    Returns
    -------
    np.ndarray
        Independent flattened snapshot, ending with six output channels.
    """
    return np.r_[
        limb.q, limb.w, limb.Limb.q_prev, limb.Limb.w_prev,
        limb.F_flex, limb.F_ext, limb.Flexor.Cn, limb.Extensor.Cn,
        limb.Flexor.x, limb.Extensor.x, limb.output,
    ]


def _make_system(system_type: Type[Net_Limb_connect]) -> Net_Limb_connect:
    """Собрать систему. Build a deterministic closed feedback loop.

    Parameters
    ----------
    system_type : type
        Net_Limb_connect or Var_Limb.

    Returns
    -------
    Net_Limb_connect
        Independently initialized two-neuron system.
    """
    net = Izhikevich_IO_Network(
        N=2, input_size=2, output_size=2, afferent_size=6,
        Q_app=np.eye(2), P=np.eye(2), W=np.zeros((2, 2)),
        Q_aff=np.array([[0.1, 0.1, 1., 0., 0., 0.],
                        [0., 0., 0., 0.1, 0.1, 1.]]),
    )
    limb = Afferented_Limb(
        Limb=OneDOFLimb(q0=1.6, w0=0.1),
        Flexor=SimpleAdaptedMuscle(N=2),
        Extensor=SimpleAdaptedMuscle(N=2),
    )
    # Старый конструктор использует глобальный RNG; тест его сохраняет.
    rng_state = np.random.get_state()
    try:
        system = system_type(Network=net, Limb=limb)
    finally:
        np.random.set_state(rng_state)
    system.set_init_conditions(v_noise=np.zeros(2))
    return system


def test_calc_afferents_base() -> None:
    """Проверить покой. Verify zero-input resting signals and finite output.

    Returns
    -------
    None
        Assertions verify resting muscle force and afferents.
    """
    limb = Afferented_Limb()
    limb.step()
    assert limb.output.shape == (6,)
    assert np.isfinite(limb.output).all()
    # cos(pi/2) is only approximately zero in floating-point arithmetic.
    np.testing.assert_allclose(limb.output, np.zeros(6), atol=1e-10)
    assert limb.F_flex == limb.F_ext == 0.0


@pytest.mark.parametrize("additional", [
    np.arange(1, 7) / 10,
    np.zeros(6),
    -np.arange(1, 7) / 10,
    None,
], ids=["positive", "zero", "negative", "none"])
def test_calc_afferents_addition(
    active_limb: Afferented_Limb, additional: Optional[np.ndarray],
) -> None:
    """Проверить сумму. Verify signed addition without clipping or aliasing.

    Parameters
    ----------
    active_limb : Afferented_Limb
        Fixture with nonzero intrinsic afferents.
    additional : np.ndarray or None
        Six signed offsets, zero input, or omitted input.

    Returns
    -------
    None
        Assertions verify every output channel and input immutability.
    """
    base = active_limb.output.copy()
    assert np.any(base > 0)
    original = None if additional is None else additional.copy()
    result = active_limb.calc_afferents(additional)
    expected = base if additional is None else base + additional
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    assert result is active_limb.output
    if additional is not None:
        np.testing.assert_array_equal(additional, original)
        assert not np.shares_memory(result, additional)
        if np.all(additional < 0):
            assert result[2] < 0  # Signed external offsets are not clipped.


@pytest.mark.parametrize("channel", range(6))
def test_channel_order(channel: int) -> None:
    """Проверить адресацию. Each offset changes exactly its own channel.

    Parameters
    ----------
    channel : int
        Index in the documented six-channel afferent order.

    Returns
    -------
    None
        Assertions check absence of channel mixing.
    """
    limb = Afferented_Limb()
    base = limb.calc_afferents().copy()
    additional = np.eye(6)[channel]
    np.testing.assert_allclose(limb.calc_afferents(additional) - base,
                               additional, atol=1e-12)


def test_repeated_calls_do_not_accumulate(
    active_limb: Afferented_Limb,
) -> None:
    """Исключить накопление. Recomputing an offset must not integrate it.

    Parameters
    ----------
    active_limb : Afferented_Limb
        Fixture with intrinsic afferents already computed.

    Returns
    -------
    None
        Checks repeated addition and restoration with None.
    """
    base = active_limb.output.copy()
    additional = np.arange(1, 7) / 10
    for _ in range(3):
        np.testing.assert_allclose(active_limb.calc_afferents(additional),
                                   base + additional)
    np.testing.assert_allclose(active_limb.calc_afferents(None), base)


def test_output_can_be_used_as_input(active_limb: Afferented_Limb) -> None:
    """Проверить общий буфер. Snapshot an input aliased to the output.

    Parameters
    ----------
    active_limb : Afferented_Limb
        Fixture used to pass the mutable output as the next input.

    Returns
    -------
    None
        Checks use of input values from before recomputation.
    """
    base = active_limb.output.copy()
    active_limb.calc_afferents(np.ones(6))
    previous = active_limb.output.copy()
    result = active_limb.calc_afferents(active_limb.output)
    np.testing.assert_allclose(result, base + previous)


def test_step_forwards_and_clears_input(active_limb: Afferented_Limb) -> None:
    """Проверить step. Forward the offset without changing muscle inputs.

    Parameters
    ----------
    active_limb : Afferented_Limb
        Fixture compared with an independent copy receiving no offset.

    Returns
    -------
    None
        Checks arithmetic, mechanical state and nonpersistent input.
    """
    reference = deepcopy(active_limb)
    additional = np.arange(1, 7) / 10
    reference.step(dt=0.1, uf=0.5, ue=0.3)
    active_limb.step(dt=0.1, uf=0.5, ue=0.3,
                     additional_afferents=additional)
    np.testing.assert_allclose(active_limb.output,
                               reference.output + additional)
    np.testing.assert_array_equal(_state(active_limb)[:-6],
                                  _state(reference)[:-6])
    active_limb.step(dt=0.1, uf=0.5, ue=0.3)
    reference.step(dt=0.1, uf=0.5, ue=0.3)
    np.testing.assert_array_equal(_state(active_limb), _state(reference))


@pytest.mark.parametrize("invalid", [
    1., np.ones(1), np.zeros(5), np.zeros(7), np.zeros((1, 6)),
    np.zeros((6, 1)), np.zeros((2, 6)), np.full(6, np.nan),
    np.full(6, np.inf), np.full(6, -np.inf), np.ones(6, dtype=complex),
])
@pytest.mark.parametrize("method", ["calc_afferents", "step", "system_step"])
def test_invalid_input_is_rejected_before_mutation(
    invalid: Any, method: str,
) -> None:
    """Отклонить неверный вход. Reject invalid vectors before state changes.

    Parameters
    ----------
    invalid : object
        Wrong shape, complex values, or nonfinite numbers.
    method : str
        Public entry point whose validation is checked.

    Returns
    -------
    None
        Assertions verify the exception and unchanged state.
    """
    system = _make_system(Var_Limb)
    before = np.r_[_state(system.Limb), system.V, system.U]
    target = system.step if method == "system_step" else getattr(
        system.Limb, method,
    )
    with pytest.raises(ValueError, match="additional_afferents"):
        target(additional_afferents=invalid)
    np.testing.assert_array_equal(
        np.r_[_state(system.Limb), system.V, system.U], before,
    )


@pytest.mark.parametrize("system_type", [Net_Limb_connect, Var_Limb])
def test_closed_loop_1000_steps(system_type: Type[Net_Limb_connect]) -> None:
    """Проверить замкнутый контур. Run 1000 steps with constant extra input.

    Parameters
    ----------
    system_type : type
        Base connection or its named Var_Limb specialization.

    Returns
    -------
    None
        Checks finite trajectories, feedback timing and changed dynamics.
    """
    system = _make_system(system_type)
    baseline = deepcopy(system)
    additional = np.array([0., 0., 2., 0., 0., 1.])
    states, references = [], []
    for index in range(1000):
        system.step(dt=0.1, Iapp=np.array([5., 5.]),
                    additional_afferents=additional)
        baseline.step(dt=0.1, Iapp=np.array([5., 5.]))
        states.append(np.r_[
            system.V, system.U, _state(system.Limb), system.I_syn.ravel(),
        ])
        references.append(np.r_[
            baseline.V, baseline.U, _state(baseline.Limb),
            baseline.I_syn.ravel(),
        ])
        if index == 0:
            # The network reads the previous step's afferent output.
            np.testing.assert_array_equal(system.V, baseline.V)
            np.testing.assert_allclose(system.Limb.output,
                                       baseline.Limb.output + additional)
        elif index == 1:
            np.testing.assert_allclose(
                system.V - baseline.V,
                0.1 * system.net.Q_aff.dot(additional), atol=1e-12,
            )
    assert np.isfinite(states).all()
    assert np.isfinite(references).all()
    assert not np.allclose(np.array(states)[:, :2],
                           np.array(references)[:, :2])
    assert not np.allclose(system.Limb.output, baseline.Limb.output)
    assert abs(system.q - baseline.q) > 1e-8


def test_reset_removes_additional_output() -> None:
    """Проверить сброс. Reset must clear the added afferent output.

    Returns
    -------
    None
        Verifies zero output and recomputed resting signals after reset.
    """
    limb = Afferented_Limb()
    limb.step(additional_afferents=np.ones(6))
    limb.set_init_conditions()
    np.testing.assert_array_equal(limb.output, np.zeros(6))
    np.testing.assert_allclose(limb.calc_afferents(), np.zeros(6), atol=1e-10)


def test_tactile_signal_channels_and_boundaries() -> None:
    """Проверить импульсы. Verify pulse channels and half-open boundaries.

    Returns
    -------
    None
        Checks overlap saturation, channel order and pulse duration.
    """
    from examples.example_additional_afferents import create_tactile_signal

    times = np.array([9., 10., 11., 12., 13., 14.])
    signal = create_tactile_signal(
        times, [10., 11.], amplitude=2., duration=2.,
    )
    np.testing.assert_array_equal(signal[:, 2], [0., 2., 2., 2., 0., 0.])
    np.testing.assert_array_equal(signal[:, 5], signal[:, 2] / 2)
    np.testing.assert_array_equal(signal[:, [0, 1, 3, 4]], np.zeros((6, 4)))


def test_example_records_matched_states_and_effects(tmp_path: Path) -> None:
    """Проверить демонстрацию. Verify matched runs and all nine plot panels.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary destination for the test figure.

    Returns
    -------
    None
        Checks determinism, causal timing, additive identity and PNG layout.
    """
    import matplotlib.pyplot as plt

    from examples.example_additional_afferents import (
        create_tactile_signal,
        plot_results,
        simulate,
    )

    # Первый контакт полного примера; ждём следующую пачку и отклик мышцы.
    times = 0.1 * np.arange(1, 40001)
    signal = create_tactile_signal(times, [2000.], amplitude=2., duration=100.)
    baseline = simulate(times, 0.1)
    repeated = simulate(times, 0.1)
    tactile = simulate(times, 0.1, signal)
    for name in baseline:
        np.testing.assert_array_equal(baseline[name], repeated[name])
        assert np.isfinite(tactile[name]).all()
    np.testing.assert_array_equal(tactile["V"][times <= 2000.],
                                  baseline["V"][times <= 2000.])
    np.testing.assert_allclose(tactile["afferents"] - tactile["intrinsic"],
                               signal, atol=1e-12)
    assert not np.allclose(tactile["V"], baseline["V"])
    assert not np.allclose(tactile["q"], baseline["q"], atol=1e-10, rtol=0)
    output = tmp_path / "additional_afferents_example.png"
    fig = plot_results({"baseline": baseline, "tactile": tactile},
                       output=output, show=False)
    try:
        np.testing.assert_array_equal(fig.get_size_inches(), [20., 12.])
        # Nine data panels plus one twin Y axis and one colorbar axis.
        assert len(fig.axes) == 11
        for axis in fig.axes[:9]:
            assert axis.get_title() and axis.get_xlabel() and axis.get_ylabel()
            assert axis.get_legend() is not None
            assert any(line.get_visible() for line in axis.get_xgridlines())
        assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert plt.imread(output).shape[:2] == (1800, 3000)
    finally:
        plt.close(fig)
