"""Behavioral checks against the unchanged core stepping interfaces."""

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from SpikingNN import ConfigurationError, Simulation, ValidationError, create_simulation, run_simulation, run_simulation_async
from SpikingNN.signals import generate_signal


@pytest.fixture
def config():
    return {
        "system_type": "onlynet",
        "network": {
            "neuron_count": 2,
            "neuron_types": ["RS", "RS"],
            "input_size": 2,
            "output_size": 2,
            "afferent_size": 0,
            "weights": [[0, 0], [0, 0]],
            "Q_app": [[0, 1], [1, 0]],
        },
        "simulation": {"dt": 0.1, "duration": 10.0},
    }


def manual_network_history(config, currents):
    net = Simulation(deepcopy(config)).network
    v, u = [], []
    for current in currents:
        v.append(net.V_prev.copy())
        u.append(net.U_prev.copy())
        net.step(dt=config["simulation"]["dt"], Iapp=current, Iaff=np.zeros(net.afferent_size))
    return np.array(v), np.array(u)


def test_empty_signal_advances_zero_current_dynamics(config):
    data = run_simulation(Simulation(config), {}).get_data()
    v, u = manual_network_history(config, np.zeros((100, 2)))
    np.testing.assert_allclose(data["V"], v)
    np.testing.assert_allclose(data["U"], u)
    assert not np.array_equal(data["V"][0], data["V"][-1])


def test_input_channel_routing_respects_q_app(config):
    signal = {"type": "constant", "amplitude": 5, "neurons": [0]}
    data = run_simulation(Simulation(config), signal).get_data()
    v, u = manual_network_history(config, np.tile([5, 0], (100, 1)))
    np.testing.assert_allclose(data["V"], v)
    np.testing.assert_allclose(data["U"], u)
    assert not np.array_equal(data["V"][:, 0], data["V"][:, 1])


def test_json_signal_default_and_explicit_override(config):
    config["simulation"]["input_current"] = {"type": "constant", "amplitude": 5}
    default = run_simulation(Simulation(config)).get_data()
    explicit = run_simulation(Simulation(config), config["simulation"]["input_current"]).get_data()
    zero = run_simulation(Simulation(config), {}).get_data()
    np.testing.assert_array_equal(default["V"], explicit["V"])
    assert not np.array_equal(default["V"], zero["V"])


def test_signal_is_generated_once_before_stepping(config):
    with patch("SpikingNN.api.generate_signal", wraps=generate_signal) as generate:
        run_simulation(Simulation(config), {"type": "constant", "amplitude": 5})
    assert generate.call_count == 1


def test_noise_samples_match_single_seeded_signal(config):
    signal = {"type": "noise", "amplitude": 1, "neurons": [1]}
    # Keep the caller's global RNG state unchanged.
    rng_state = np.random.get_state()
    try:
        np.random.seed(42)
        values = generate_signal(signal, duration=10, dt=0.1)
        np.random.seed(42)
        data = run_simulation(Simulation(config), signal).get_data()
    finally:
        np.random.set_state(rng_state)
    currents = np.column_stack([np.zeros(100), values])
    v, u = manual_network_history(config, currents)
    np.testing.assert_allclose(data["V"], v)
    np.testing.assert_allclose(data["U"], u)


@pytest.mark.parametrize("signal, field", [
    ({"type": "unknown", "amplitude": 1}, "type"),
    ({"type": "constant"}, "amplitude"),
    ({"type": "constant", "amplitude": float("nan")}, "amplitude"),
    ({"type": "constant", "amplitude": "5"}, "amplitude"),
    ({"type": "sine", "amplitude": 1, "frequency": -1}, "frequency"),
    ({"type": "noise", "amplitude": -1}, "amplitude"),
    ({"type": "constant", "amplitude": 1, "neurons": [0.5]}, "neurons"),
    ({"type": "constant", "amplitude": 1, "neurons": [2]}, "neurons"),
    ({"type": "constant", "amplitude": 1, "phase": float("inf")}, "phase"),
])
def test_invalid_signal_fails_before_mutating_state(config, signal, field):
    sim = Simulation(config)
    before = sim.network.V_prev.copy()
    with pytest.raises(ValidationError) as exc:
        run_simulation(sim, signal)
    assert exc.value.field == field
    np.testing.assert_array_equal(sim.network.V_prev, before)


def full_config(config):
    config = deepcopy(config)
    config["system_type"] = "fullsys"
    config["network"].update({
        "output_size": 4, "afferent_size": 12,
        "P": [[1, 0], [0, 0], [0, 0], [0, 1]],
        "Q_aff": np.zeros((2, 12)).tolist(),
    })
    config["network"]["Q_aff"][0][0] = 0.01
    config["network"]["Q_aff"][1][6] = 0.01
    config["limbs"] = [
        {"name": "left", "mechanics": {"q0": 1.2}},
        {"name": "right", "mechanics": {"q0": 1.8}},
    ]
    return config


def test_fullsys_records_all_limbs_and_afferent_snapshots(config):
    config = full_config(config)
    actual = run_simulation(Simulation(config), {}).get_data()
    reference = Simulation(config)
    states = []
    for _ in actual["time"]:
        states.append(reference.system.get_state())
        reference.system.step(dt=0.1, Iapp=np.zeros(2))
    np.testing.assert_allclose(actual["V"], [state["neurons_V"] for state in states])
    np.testing.assert_allclose(actual["U"], [state["neurons_U"] for state in states])
    assert set(actual["limbs"]) == {"left", "right"}
    for name in actual["limbs"]:
        for key, values in actual["limbs"][name].items():
            expected = np.array([state["limbs"][name][key] for state in states])
            np.testing.assert_allclose(values, expected)
        assert actual["limbs"][name]["afferents"].shape == (100, 6)
        assert not np.array_equal(actual["limbs"][name]["angle_q"][0], actual["limbs"][name]["angle_q"][-1])


@pytest.mark.parametrize("names", [["same", "same"], ["limb_1", None]])
def test_fullsys_rejects_duplicate_effective_limb_names(config, names):
    config = full_config(config)
    for limb, name in zip(config["limbs"], names):
        if name is None:
            limb.pop("name")
        else:
            limb["name"] = name
    with pytest.raises(ConfigurationError):
        Simulation(config)


def test_fullsys_requires_limbs(config):
    config["system_type"] = "fullsys"
    with pytest.raises(ConfigurationError):
        Simulation(config)


def test_repeated_run_continues_current_state_and_preserves_previous_results(config):
    sim = Simulation(config)
    first = run_simulation(sim, {}).get_data()
    saved = first["V"].copy()
    final_state = sim.network.V_prev.copy()
    second = run_simulation(sim, {}).get_data()
    np.testing.assert_array_equal(second["V"][0], final_state)
    np.testing.assert_array_equal(first["V"], saved)


def test_configuration_is_owned_by_simulation(config):
    sim = Simulation(config)
    config["simulation"]["dt"] = -1
    assert sim.config["simulation"]["dt"] == 0.1


@pytest.mark.parametrize("fullsys", [False, True])
def test_progress_is_optional_and_preserves_results(config, capsys, fullsys):
    if fullsys:
        config = full_config(config)
    quiet = run_simulation(Simulation(config), {}).get_data()
    assert capsys.readouterr().err == ""
    visible = run_simulation(Simulation(config), {}, progress=True).get_data()
    output = capsys.readouterr()
    assert "100%" in output.err
    assert "100/100" in output.err
    for key in ("time", "V", "U"):
        np.testing.assert_array_equal(visible[key], quiet[key])
    if fullsys:
        for name, history in quiet["limbs"].items():
            for key, values in history.items():
                np.testing.assert_array_equal(visible["limbs"][name][key], values)


@pytest.mark.asyncio
async def test_async_progress(config, capsys):
    result = await run_simulation_async(Simulation(config), {}, progress=True)
    expected = run_simulation(Simulation(config), {}).get_data()
    np.testing.assert_array_equal(result.get_data()["V"], expected["V"])
    assert "100/100" in capsys.readouterr().err


def test_json_neuron_parameters_override_presets_and_initial_conditions(config, tmp_path):
    import json

    config["network"]["neuron_types"] = ["CH", "FS"]
    parameters = {"a": [0.003, 0.1], "b": [0.2, 0.2], "c": [-55, -65], "d": [2, 2]}
    config["network"]["neuron_params"] = parameters
    path = tmp_path / "manual-neurons.json"
    path.write_text(json.dumps(config))
    simulation = create_simulation(path)
    for name, expected in parameters.items():
        np.testing.assert_array_equal(getattr(simulation.network, name), expected)
    np.testing.assert_array_equal(simulation.network.V_prev, parameters["c"])
    np.testing.assert_allclose(simulation.network.U_prev, np.array(parameters["b"]) * parameters["c"])
    data = run_simulation(simulation, {"type": "constant", "amplitude": 8}).get_data()
    assert np.isfinite(data["V"]).all()
    v, u = manual_network_history(config, np.full((100, 2), 8))
    np.testing.assert_array_equal(data["V"], v)
    np.testing.assert_array_equal(data["U"], u)


def test_partial_neuron_parameters_preserve_other_presets(config):
    config["network"]["neuron_types"] = ["CH", "FS"]
    original = Simulation(config)
    config["network"]["neuron_params"] = {"a": [0.003, 0.1]}
    custom = Simulation(config)
    np.testing.assert_array_equal(custom.network.a, [0.003, 0.1])
    for name in ("b", "c", "d"):
        np.testing.assert_array_equal(getattr(custom.network, name), getattr(original.network, name))
    signal = {"type": "constant", "amplitude": 8}
    default_v = run_simulation(original, signal).get_data()["V"]
    custom_v = run_simulation(custom, signal).get_data()["V"]
    assert not np.array_equal(default_v[:, 0], custom_v[:, 0])
    np.testing.assert_array_equal(default_v[:, 1], custom_v[:, 1])


@pytest.mark.parametrize("parameters", [
    {"a": [0.003]},
    {"a": [0.003, float("nan")]},
    {"a": [0.003, float("inf")]},
    {"a": 0.003},
    {"unknown": [0.003, 0.1]},
])
def test_invalid_neuron_parameters_are_rejected(config, parameters):
    config["network"]["neuron_params"] = parameters
    with pytest.raises(ConfigurationError):
        Simulation(config)
