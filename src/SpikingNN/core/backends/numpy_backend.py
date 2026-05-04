# src/SpikingNN/core/backends/numpy_backend.py
"""
NumPy backend implementation for SpikingNN computations.
Реализация бэкенда на NumPy для вычислений в SpikingNN.

This module provides a NumPy-based implementation of the Backend abstract class.
It is the default backend, offering good performance and wide compatibility.

Данный модуль предоставляет реализацию абстрактного класса Backend на базе NumPy.
Это бэкенд по умолчанию, обеспечивающий хорошую производительность и широкую совместимость.
"""
import numpy as np
from typing import Tuple
from .base import Backend


class NumPyBackend(Backend):
    """
    NumPy-based computational backend.
    Вычислительный бэкенд на базе NumPy.

    Uses standard NumPy operations for all array computations.
    Suitable for CPU-based simulations with moderate network sizes.

    Использует стандартные операции NumPy для всех вычислений с массивами.
    Подходит для симуляций на CPU с сетями умеренного размера.
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
        # Vectorized Izhikevich dynamics
        dVdt = 0.04 * np.power(V, 2) + 5 * V + 140 - U + I_app + np.sum(I_syn, axis=1)
        dUdt = a * (b * V - U)
        return dVdt, dUdt

    def where(self, condition, x, y) -> np.ndarray:
        return np.where(condition, x, y)

    def sum(self, arr, axis=None) -> np.ndarray:
        return np.sum(arr, axis=axis)

    def dot(self, a, b) -> np.ndarray:
        return np.dot(a, b)