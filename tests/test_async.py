import asyncio
import json
import tempfile
import pytest
from SpikingNN.api import create_simulation, run_simulation_async, get_results

@pytest.mark.asyncio
async def test_run_simulation_async():
    config = {
        "network": {"neuron_count": 5, "neuron_types": ["RS"]*5},
        "simulation": {"dt": 0.1, "duration": 10.0}
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
    sim = create_simulation(f.name)
    results = await run_simulation_async(sim, {})
    data = get_results(results)
    assert "V" in data
    assert "U" in data
    assert data["V"].shape == (100, 5)
