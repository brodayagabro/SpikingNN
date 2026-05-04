# src/SpikingNN/core/backends/torch_backend.py
"""
PyTorch backend implementation for GPU-accelerated SpikingNN computations.
Реализация бэкенда на PyTorch для вычислений с ускорением на GPU.

This module provides a PyTorch-based implementation supporting CUDA acceleration.
Ideal for very large networks or batch simulations on GPU hardware.

Данный модуль предоставляет реализацию на базе PyTorch с поддержкой ускорения CUDA.
Идеален для очень больших сетей или пакетных симуляций на оборудовании с GPU.
"""
import numpy as np
from typing import Tuple, Union
import torch
from .base import Backend

ArrayLike = Union[np.ndarray, torch.Tensor]


class TorchBackend(Backend):
    """
    PyTorch-based computational backend with optional CUDA support.
    Вычислительный бэкенд на базе PyTorch с опциональной поддержкой CUDA.

    Automatically detects available GPU and moves tensors accordingly.
    All operations use PyTorch tensor API for gradient tracking compatibility.

    Автоматически обнаруживает доступный GPU и перемещает тензоры соответствующим образом.
    Все операции используют API тензоров PyTorch для совместимости с отслеживанием градиентов.
    """

    def __init__(self, device: str = None):
        """
        Initialize TorchBackend with specified device.

        Parameters
        ----------
        device : str, optional
            'cuda', 'cpu', or None (auto-detect).

        Инициализировать TorchBackend с указанным устройством.

        Параметры
        ----------
        device : str, optional
            'cuda', 'cpu' или None (автоопределение).
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

    def _to_tensor(self, data, dtype=None) -> torch.Tensor:
        """Helper to convert data to torch tensor on correct device."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        return torch.tensor(data, dtype=dtype, device=self.device)

    def array(self, data, dtype=None) -> torch.Tensor:
        return self._to_tensor(data, dtype)

    def zeros(self, shape, dtype=float) -> torch.Tensor:
        torch_dtype = torch.float32 if dtype == float else torch.float64
        return torch.zeros(shape, dtype=torch_dtype, device=self.device)

    def run_state(
        self,
        V: torch.Tensor,
        U: torch.Tensor,
        I_syn: torch.Tensor,
        I_app: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Vectorized Izhikevich dynamics using PyTorch ops
        dVdt = 0.04 * torch.pow(V, 2) + 5 * V + 140 - U + I_app + torch.sum(I_syn, dim=1)
        dUdt = a * (b * V - U)
        return dVdt, dUdt

    def where(self, condition, x, y) -> torch.Tensor:
        return torch.where(condition, x, y)

    def sum(self, arr, axis=None) -> torch.Tensor:
        return torch.sum(arr, dim=axis)

    def dot(self, a, b) -> torch.Tensor:
        return torch.matmul(a, b)