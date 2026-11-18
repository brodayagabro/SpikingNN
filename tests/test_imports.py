from SpikingNN import create_simulation, run_simulation, get_results
from importlib import import_module
import pytest

def test_api_imports():
    assert callable(create_simulation)
    assert callable(run_simulation)
    assert callable(get_results)


@pytest.mark.parametrize("module", [
    "SpikingNN.Networks.Rybak2002",
    "SpikingNN.sys_output",
    "apps.streamlit_app.core.io_json",
])
def test_core_consumers_import_without_path_workarounds(module):
    import_module(module)
