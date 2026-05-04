# src/SpikingNN/core/backends/base.py
"""
Abstract backend interface for spiking neural network computations.
Абстрактный интерфейс бэкенда для вычислений в спайковых нейронных сетях.

This module defines the base class that all computational backends must inherit.
It ensures a consistent API across NumPy, Numba, and PyTorch implementations.

Данный модуль определяет базовый класс, от которого должны наследоваться
все вычислительные бэкенды. Это обеспечивает единый API для реализаций
на NumPy, Numba и PyTorch.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np


class Backend(ABC):
    """
    Abstract base class for computational backends.
    Абстрактный базовый класс для вычислительных бэкендов.

    All backends must implement methods for:
    - Array creation and manipulation
    - Izhikevich neuron dynamics (run_state)
    - Threshold detection and reset logic
    - Synaptic current updates

    Все бэкенды должны реализовывать методы для:
    - Создания и манипуляции массивами
    - Динамики нейрона Ижикеича (run_state)
    - Детекции порога и логики сброса
    - Обновления синаптического тока
    """

    @abstractmethod
    def array(self, data, dtype=None) -> np.ndarray:
        """
        Create an array from input data.

        Parameters
        ----------
        data : array-like
            Input data to convert to array.
        dtype : data-type, optional
            Desired data type of the array.

        Returns
        -------
        ndarray
            Array in backend-specific format.

        Создать массив из входных данных.

        Параметры
        ----------
        data : array-like
            Входные данные для преобразования в массив.
        dtype : data-type, optional
            Желаемый тип данных массива.

        Возвращает
        ----------
        ndarray
            Массив в формате, специфичном для бэкенда.
        """
        pass

    @abstractmethod
    def zeros(self, shape, dtype=float) -> np.ndarray:
        """
        Create an array filled with zeros.

        Parameters
        ----------
        shape : int or tuple of ints
            Shape of the new array.
        dtype : data-type, optional
            Desired data type.

        Returns
        -------
        ndarray
            Zero-filled array.

        Создать массив, заполненный нулями.

        Параметры
        ----------
        shape : int или кортеж int
            Форма нового массива.
        dtype : data-type, optional
            Желаемый тип данных.

        Возвращает
        ----------
        ndarray
            Массив, заполненный нулями.
        """
        pass

    @abstractmethod
    def run_state(
        self,
        V: np.ndarray,
        U: np.ndarray,
        I_syn: np.ndarray,
        I_app: np.ndarray,
        a: np.ndarray,
        b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute derivatives for Izhikevich neuron dynamics.

        Implements: dV/dt = 0.04*V² + 5*V + 140 - U + I_app + ΣI_syn
                    dU/dt = a*(b*V - U)

        Parameters
        ----------
        V : ndarray
            Membrane potentials (mV).
        U : ndarray
            Recovery variables.
        I_syn : ndarray
            Synaptic currents matrix (N×N).
        I_app : ndarray
            Applied input currents.
        a, b : ndarray
            Izhikevich parameters.

        Returns
        -------
        tuple of ndarray
            (dVdt, dUdt) derivatives.

        Вычислить производные для динамики нейрона Ижикеича.

        Реализует: dV/dt = 0.04*V² + 5*V + 140 - U + I_app + ΣI_syn
                   dU/dt = a*(b*V - U)

        Параметры
        ----------
        V : ndarray
            Мембранные потенциалы (мВ).
        U : ndarray
            Переменные восстановления.
        I_syn : ndarray
            Матрица синаптических токов (N×N).
        I_app : ndarray
            Приложенные входные токи.
        a, b : ndarray
            Параметры Ижикеича.

        Возвращает
        -------
        tuple of ndarray
            (dVdt, dUdt) производные.
        """
        pass

    @abstractmethod
    def where(self, condition, x, y) -> np.ndarray:
        """
        Element-wise conditional selection.

        Parameters
        ----------
        condition : array-like of bool
            Where True, yield x, otherwise yield y.
        x, y : array-like
            Values to select from.

        Returns
        -------
        ndarray
            Result array with selected elements.

        Поэлементный условный выбор.

        Параметры
        ----------
        condition : array-like of bool
            Где True, выбрать x, иначе выбрать y.
        x, y : array-like
            Значения для выбора.

        Возвращает
        -------
        ndarray
            Результирующий массив с выбранными элементами.
        """
        pass

    @abstractmethod
    def sum(self, arr, axis=None) -> np.ndarray:
        """
        Sum array elements over given axis.

        Parameters
        ----------
        arr : ndarray
            Input array.
        axis : int or tuple of ints, optional
            Axis or axes along which to sum.

        Returns
        -------
        ndarray
            Summed array.

        Суммировать элементы массива по заданной оси.

        Параметры
        ----------
        arr : ndarray
            Входной массив.
        axis : int или кортеж int, optional
            Ось или оси, по которым суммировать.

        Возвращает
        -------
        ndarray
            Массив с суммами.
        """
        pass

    @abstractmethod
    def dot(self, a, b) -> np.ndarray:
        """
        Matrix multiplication.

        Parameters
        ----------
        a, b : ndarray
            Input matrices.

        Returns
        -------
        ndarray
            Matrix product.

        Умножение матриц.

        Параметры
        ----------
        a, b : ndarray
            Входные матрицы.

        Возвращает
        -------
        ndarray
            Произведение матриц.
        """
        pass