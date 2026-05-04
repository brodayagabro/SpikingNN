# src/SpikingNN/core/backends/numba_backend.py
"""
Numba JIT backend implementation for accelerated SpikingNN computations.
Реализация бэкенда с JIT-компиляцией через Numba для ускоренных вычислений.

This module provides a Numba-accelerated implementation using @njit decorator.
Ideal for large-scale simulations requiring high performance on CPU.

Данный модуль предоставляет реализацию с ускорением через Numba с использованием
декоратора @njit. Идеален для крупномасштабных симуляций, требующих высокой
производительности на CPU.
"""
import numpy as np
from typing import Tuple
from numba import njit, prange
from .base import Backend


@njit(parallel=True)
def _run_state_numba(
    V: np.ndarray,
    U: np.ndarray,
    I_syn: np.ndarray,
    I_app: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    N: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-compiled Izhikevich dynamics.
    Динамика Ижикеича, скомпилированная через Numba.
    """
    dVdt = np.zeros(N)
    dUdt = np.zeros(N)
    for i in prange(N):
        I_syn_sum = 0.0
        for j in range(N):
            I_syn_sum += I_syn[j, i]
        dVdt[i] = 0.04 * V[i] * V[i] + 5 * V[i] + 140 - U[i] + I_app[i] + I_syn_sum
        dUdt[i] = a[i] * (b[i] * V[i] - U[i])
    return dVdt, dUdt


class NumbaBackend(Backend):
    """
    Numba JIT-accelerated computational backend.
    Вычислительный бэкенд с JIT-ускорением через Numba.

    Uses @njit compilation for critical loops, providing 10-100x speedup
    for large networks compared to pure NumPy.

    Использует компиляцию @njit для критических циклов, обеспечивая ускорение
    в 10-100 раз для больших сетей по сравнению с чистым NumPy.
    """

    def array(self, data, dtype=None) -> np.ndarray:
        return np.array(data, dtype=dtype)

    def zeros(self, shape, dtype=float) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    def run_state(
        self,
        V: np.ndarray,
        U: np.ndarray,
        I_syn: np.ndarray,
        I_app: np.ndarray,
        a: np.ndarray,
        b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        N = len(V)
        return _run_state_numba(V, U, I_syn, I_app, a, b, N)

    def where(self, condition, x, y) -> np.ndarray:
        return np.where(condition, x, y)

    def sum(self, arr, axis=None) -> np.ndarray:
        return np.sum(arr, axis=axis)

    def dot(self, a, b) -> np.ndarray:
        return np.dot(a, b)