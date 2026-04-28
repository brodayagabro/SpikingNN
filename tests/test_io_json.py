"""
Tests for JSON import/export functionality using Rybak2002 configuration.
"""
import sys
import os
import pytest
import numpy as np
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from streamlit_app.core.io_json import (
    system_to_dict, dict_to_system, save_system_json, load_system_json,
    _numpy_to_list, _list_to_numpy, _serialize_network, 
    _serialize_afferented_limb, _deserialize_afferented_limb
)
from src.SpikingNN.Izh_net import (
    Izhikevich_IO_Network, Afferented_Limb, SimpleAdaptedMuscle, OneDOFLimb
)
from src.SpikingNN.Networks.Rybak2002 import Rybak_2002_network


class TestNumpySerialization:
    def test_numpy_to_list_preserves_data(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        res = _numpy_to_list(arr)
        assert res['dtype'] == 'float64' and res['shape'] == (2, 2)
        
    def test_list_to_numpy_roundtrip(self):
        original = np.random.rand(4, 6).astype(np.float32)
        restored = _list_to_numpy(_numpy_to_list(original))
        assert np.allclose(restored, original)


class TestNetworkSerialization:
    def test_rybak_network_serialization(self):
        net = Rybak_2002_network(output_size=2, input_size=2)
        d = _serialize_network(net)
        assert d['N'] == 12 and d['afferent_size'] == 6
        
    def test_network_roundtrip(self):
        from streamlit_app.core.io_json import _deserialize_network
        net_orig = Rybak_2002_network(output_size=2, input_size=2)
        net_rest = _deserialize_network(_serialize_network(net_orig))
        assert np.allclose(net_rest.W, net_orig.W)


class TestLimbSerialization:
    def test_afferented_limb_serialization(self):
        limb = OneDOFLimb(m=0.3, ls=0.3, b=0.002, a1=0.4, a2=0.05)
        al = Afferented_Limb(Limb=limb, Flexor=SimpleAdaptedMuscle(w=0.5), 
                             Extensor=SimpleAdaptedMuscle(w=0.4))
        d = _serialize_afferented_limb(al)
        assert d['limb']['m'] == 0.3 and d['flexor']['w'] == 0.5
        
    def test_limb_roundtrip(self):
        al_orig = Afferented_Limb(
            Limb=OneDOFLimb(m=0.25, ls=0.35), 
            Flexor=SimpleAdaptedMuscle(w=0.55, tau_c=1/65), 
            Extensor=SimpleAdaptedMuscle(w=0.45, tau_1=1/120)
        )
        al_rest = _deserialize_afferented_limb(_serialize_afferented_limb(al_orig))
        assert al_rest.Limb.m == 0.25


class TestFullSystemIO:
    def test_rybak_system_save_load(self, tmp_path):
        from src.SpikingNN.Var_Limb import Var_Limb
        net = Rybak_2002_network(output_size=2, input_size=2)
        limb = Afferented_Limb()
        system = Var_Limb(Network=net, Limb=limb)
        
        filepath = tmp_path / "rybak_config.json"
        save_system_json(system, filepath)
        
        assert filepath.exists()
        with open(filepath, 'r') as f:
            config = json.load(f)
        assert 'network' in config and 'limb' in config
        
        sys_loaded = load_system_json(filepath, system_class=Var_Limb)
        assert sys_loaded.net.N == 12
        assert np.allclose(sys_loaded.net.W, system.net.W)
        
    def test_system_step_consistency_after_load(self, tmp_path):
        from src.SpikingNN.Var_Limb import Var_Limb
        net = Rybak_2002_network(output_size=2, input_size=2)
        limb = Afferented_Limb()
        system_orig = Var_Limb(Network=net, Limb=limb)
        
        rng = np.random.RandomState(42)
        v_noise = rng.normal(size=12, scale=0.5)
        q0, w0 = np.pi/2, 0.0
        system_orig.set_init_conditions(v_noise=v_noise, q0=q0, w0=w0)
        
        dt, Iapp = 0.1, np.array([5.0, 5.0])
        system_orig.step(dt=dt, Iapp=Iapp)
        state_ref = (system_orig.q, system_orig.w, system_orig.V.copy(), system_orig.F_flex)
        
        filepath = tmp_path / "test_system.json"
        save_system_json(system_orig, filepath)
        sys_loaded = load_system_json(filepath, system_class=Var_Limb)
        sys_loaded.set_init_conditions(v_noise=v_noise, q0=q0, w0=w0)
        sys_loaded.step(dt=dt, Iapp=Iapp)
        
        q, w, V, F = sys_loaded.q, sys_loaded.w, sys_loaded.V, sys_loaded.F_flex
        assert np.isclose(q, state_ref[0], atol=1e-10)
        assert np.isclose(w, state_ref[1], atol=1e-10)
        assert np.allclose(V, state_ref[2], atol=1e-10)
        assert np.isclose(F, state_ref[3], atol=1e-10)


class TestEdgeCases:
    @pytest.mark.skipif(sys.platform == 'win32', reason="Windows path permissions differ")
    def test_invalid_json_path(self):
        from src.SpikingNN.Var_Limb import Var_Limb
        net = Rybak_2002_network(output_size=2, input_size=2)
        with pytest.raises((PermissionError, OSError)):
            save_system_json(Var_Limb(Network=net), "/root/protected/config.json")
            
    def test_corrupted_json_load(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{ invalid json")
        with pytest.raises(json.JSONDecodeError):
            load_system_json(p)
            
    def test_missing_required_fields(self):
        with pytest.raises(ValueError):
            dict_to_system({'metadata': {'version': '1.0'}})