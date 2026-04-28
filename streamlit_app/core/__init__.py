"""
Core module for neuromechanical system utilities.
"""

from .io_json import (
    system_to_dict,
    dict_to_system,
    save_system_json,
    load_system_json,
    _numpy_to_list,
    _list_to_numpy
)

__all__ = [
    'system_to_dict',
    'dict_to_system', 
    'save_system_json',
    'load_system_json',
    '_numpy_to_list',
    '_list_to_numpy'
]