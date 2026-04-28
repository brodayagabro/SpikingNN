"""
Optimized simulation dynamics engine for Net_Limb_connect and Var_Limb systems.
Pre-allocates numpy arrays to avoid list.append overhead.
Fixed: Guaranteed saving of scalar variables (F_flex, F_ext, q, w) and robust type handling.
"""

import numpy as np
from typing import Callable, Dict, List, Any, Union


def run_simulation(
    system: Any, 
    T: np.ndarray, 
    Iapp: Callable[[float], Union[float, np.ndarray]],
    record_vars: List[str] = None
) -> Dict[str, np.ndarray]:
    """
    Execute forward-time simulation of a neuromechanical system with pre-allocated memory.
    
    Iterates through time array T, applies external input Iapp(t) at each step,
    and records specified state variables. Handles scalars and vectors by ensuring
    all recorded values are at least 1-dimensional. Pre-allocates numpy arrays 
    to eliminate Python list overhead and final concatenation.
    
    Выполняет прямое моделирование нейромеханической системы с предвыделением памяти.
    Проходит по массиву времени T, применяет внешний ввод Iapp(t) на каждом шаге
    и записывает указанные переменные состояния. Гарантирует корректную запись 
    скаляров и векторов. Предвыделяет numpy-массивы для устранения накладных расходов 
    списков Python и финальной конкатенации.
    
    Parameters
    ----------
    system : Any
        Var_Limb or Net_Limb_connect instance.
        Экземпляр класса Var_Limb или Net_Limb_connect.
    T : np.ndarray
        1D array of time points in milliseconds.
        Одномерный массив точек времени в миллисекундах.
    Iapp : callable
        Function Iapp(t) returning scalar or array.
        Функция Iapp(t), возвращающая скаляр или массив.
    record_vars : list of str, optional
        Variable names to record. Defaults to 
        ["V", "q", "w", "F_flex", "F_ext", "V_out", "output"].
        Имена переменных для записи. По умолчанию 
        ["V", "q", "w", "F_flex", "F_ext", "V_out", "output"].
        
    Returns
    -------
    dict
        Dictionary mapping variable names to 2D numpy arrays of shape (len(T), dim).
        Словарь, сопоставляющий имена переменных с 2D numpy-массивами формы (len(T), dim).
    """
    if len(T) < 2:
        raise ValueError("Time array T must contain at least 2 time points.")
        
    dt = T[1] - T[0]
    if dt <= 0:
        raise ValueError("Time step dt must be positive.")
        
    if record_vars is None:
        record_vars = ["V", "q", "w", "F_flex", "F_ext", "V_out", "output"]
        
    n_steps = len(T)
    results: Dict[str, np.ndarray] = {"T": T}
    buffers: Dict[str, np.ndarray] = {}
    valid_vars: List[str] = []
    
    # 1. Probe shapes & pre-allocate arrays
    # Определяем формы переменных на t=0 и сразу выделяем память
    for var in record_vars:
        try:
            val = getattr(system, var)
            # np.atleast_1d guarantees shape (1,) for scalars, preserving shape for arrays
            arr = np.atleast_1d(np.asarray(val, dtype=float))
            buffers[var] = np.zeros((n_steps, arr.shape[0]), dtype=float)
            valid_vars.append(var)
        except AttributeError:
            continue
            
    # 2. Main simulation loop
    for i, t in enumerate(T):
        for var in valid_vars:
            # Direct getattr is fast enough; caching adds complexity with @property
            arr = np.atleast_1d(np.asarray(getattr(system, var), dtype=float))
            buffers[var][i] = arr
            
        system.step(dt=dt, Iapp=Iapp(t))
        
    results.update(buffers)
    return results