"""
JSON Import/Export module for Net_Limb_connect neuromechanical system.
"""

import json
import numpy as np
from typing import Union, Dict, Any, Optional
from pathlib import Path

# Используем абсолютные импорты для согласованности с pytest и структурой проекта
from SpikingNN.Izh_net import (
    Izhikevich_IO_Network,
    Afferented_Limb,
    SimpleAdaptedMuscle,
    OneDOFLimb,
    Afferents
)

from SpikingNN.Var_Limb import Var_Limb


def _numpy_to_list(arr: np.ndarray) -> dict:
    return {
        'data': arr.tolist(),
        'dtype': str(arr.dtype),
        'shape': arr.shape
    }


def _list_to_numpy(data: dict) -> np.ndarray:
    return np.array(data['data'], dtype=data['dtype']).reshape(data['shape'])


def _serialize_network(net: Izhikevich_IO_Network) -> Dict[str, Any]:
    return {
        'N': net.N,
        'input_size': net.input_size,
        'output_size': net.output_size,
        'afferent_size': net.afferent_size,
        'names': net.names,
        'a': _numpy_to_list(net.a),
        'b': _numpy_to_list(net.b),
        'c': _numpy_to_list(net.c),
        'd': _numpy_to_list(net.d),
        'V_peak': net.V_peak,
        'M': _numpy_to_list(net.M),
        'W': _numpy_to_list(net.W),
        'tau_syn': _numpy_to_list(1.0 / net.tau_syn),
        'Q_app': _numpy_to_list(net.Q_app),
        'Q_aff': _numpy_to_list(net.Q_aff),
        'P': _numpy_to_list(net.P)
    }


def _deserialize_network(data: Dict[str, Any]) -> Izhikevich_IO_Network:
    kwargs = {
        'N': data['N'], 'input_size': data['input_size'],
        'output_size': data['output_size'], 'afferent_size': data['afferent_size'],
        'names': data['names'], 'a': _list_to_numpy(data['a']),
        'b': _list_to_numpy(data['b']), 'c': _list_to_numpy(data['c']),
        'd': _list_to_numpy(data['d']), 'M': _list_to_numpy(data['M']),
        'W': _list_to_numpy(data['W']), 'tau_syn': 1.0 / _list_to_numpy(data['tau_syn']),
        'Q_app': _list_to_numpy(data['Q_app']), 'Q_aff': _list_to_numpy(data['Q_aff']),
        'P': _list_to_numpy(data['P'])
    }
    net = Izhikevich_IO_Network(**kwargs)
    net.V_peak = data.get('V_peak', 30.0)
    return net


def _serialize_muscle(muscle: SimpleAdaptedMuscle) -> Dict[str, Any]:
    return {
        'w': muscle.w, 'A': muscle.A, 'N': muscle.N,
        'tau_c': 1.0 / muscle.tau_c, 'tau_1': 1.0 / muscle.tau_1,
        'm': muscle.m, 'k': muscle.k
    }


def _deserialize_muscle(data: Dict[str, Any]) -> SimpleAdaptedMuscle:
    return SimpleAdaptedMuscle(
        w=data['w'], A=data['A'], N=data['N'],
        tau_c=1.0/data['tau_c'], tau_1=1.0/data['tau_1'],
        m=data['m'], k=data['k']
    )


def _serialize_limb(limb: OneDOFLimb) -> Dict[str, Any]:
    return {
        'm': limb.m, 'ls': limb.ls, 'b': limb.b,
        'a1': limb.a1, 'a2': limb.a2, 'q0': limb.q0,
        'w0': limb.w0, 'g': limb.g
    }


def _deserialize_limb(data: Dict[str, Any]) -> OneDOFLimb:
    limb = OneDOFLimb(
        m=data['m'], ls=data['ls'], b=data['b'],
        a1=data['a1'], a2=data['a2'], q0=data.get('q0', np.pi/2),
        w0=data.get('w0', 0.0)
    )
    limb.g = data.get('g', 9.81)
    return limb


def _serialize_afferents(afferents: Afferents) -> Dict[str, Any]:
    return {k: getattr(afferents, k) for k in [
        'p_v', 'k_v', 'k_dI', 'k_dII', 'k_nI', 'k_nII', 
        'k_f', 'L_th', 'F_th', 'const_I', 'const_II'
    ]}


def _deserialize_afferents(data: Dict[str, Any]) -> Afferents:
    aff = Afferents()
    for k, v in data.items():
        if hasattr(aff, k): setattr(aff, k, v)
    return aff


def _serialize_afferented_limb(al: Afferented_Limb) -> Dict[str, Any]:
    return {
        'limb': _serialize_limb(al.Limb),
        'flexor': _serialize_muscle(al.Flexor),
        'extensor': _serialize_muscle(al.Extensor),
        'afferents': _serialize_afferents(al.Afferents)
    }


def _deserialize_afferented_limb(data: Dict[str, Any]) -> Afferented_Limb:
    limb = _deserialize_limb(data['limb'])
    flexor = _deserialize_muscle(data['flexor'])
    extensor = _deserialize_muscle(data['extensor'])
    afferents = _deserialize_afferents(data['afferents'])
    
    al = Afferented_Limb(Limb=limb, Flexor=flexor, Extensor=extensor)
    al.Afferents = afferents
    al.Afferents.L_th = np.sqrt(limb.a1**2 + limb.a2**2)
    return al


def system_to_dict(system: Any) -> Dict[str, Any]:
    result = {
        'metadata': {
            'version': '1.0',
            'system_type': type(system).__name__,
            'description': 'Neuromechanical system configuration'
        }
    }
    # Duck-typed serialization to avoid isinstance failures across import paths
    if hasattr(system, 'net') and hasattr(system.net, 'N'):
        try:
            result['network'] = _serialize_network(system.net)
        except Exception as e:
            print(f"Warning: Network serialization skipped: {e}")
            
    if hasattr(system, 'Limb') and hasattr(system.Limb, 'Flexor'):
        try:
            result['limb'] = _serialize_afferented_limb(system.Limb)
        except Exception as e:
            print(f"Warning: Limb serialization skipped: {e}")
            
    return result


def dict_to_system(data: Dict[str, Any], system_class=Var_Limb) -> Any:
    net = _deserialize_network(data['network']) if 'network' in data else None
    limb = _deserialize_afferented_limb(data['limb']) if 'limb' in data else None
    
    if net is not None and limb is not None:
        return system_class(Network=net, Limb=limb)
    elif net is not None:
        return system_class(Network=net)
    elif limb is not None:
        return limb
    else:
        raise ValueError("Configuration must contain at least 'network' or 'limb' data")


def save_system_json(system: Any, filepath: Union[str, Path], 
                    indent: int = 2, ensure_ascii: bool = False) -> None:
    config = system_to_dict(system)
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=indent, ensure_ascii=ensure_ascii)


def load_system_json(filepath: Union[str, Path], 
                    system_class=Var_Limb) -> Any:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
    with open(filepath, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    return dict_to_system(config, system_class=system_class)