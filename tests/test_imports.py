from SpikingNN import create_simulation, run_simulation, get_results

def test_api_imports():
    assert callable(create_simulation)
    assert callable(run_simulation)
    assert callable(get_results)