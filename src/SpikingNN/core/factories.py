"""
Модуль фабрик для создания компонентов SpikingNN.
Module containing factories for creating SpikingNN components.
"""

import json
import os
import numpy as np
from typing import Dict, Any, Union
from abc import ABC, abstractmethod

# Импорт необходимых классов из ядра
from SpikingNN.core.Izh_net import (
    Izhikevich_IO_Network,
    types2params,
    SimpleAdaptedMuscle,
    OneDOFLimb,
    OneDOFLimb_withGR,
    Afferented_Limb
)


class BaseFactory(ABC):
    """
    Базовый абстрактный класс для всех фабрик конфигураций.
    
    Base abstract class for all configuration factories.
    """
    
    def __init__(self):
        self.config_cache = {}

    def load_config(self, source: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Универсальный загрузчик конфигурации.
        
        Universal config loader. Accepts either a file path or a dictionary.
        """
        if isinstance(source, dict):
            return source.copy()
        
        elif isinstance(source, str):
            if source in self.config_cache:
                return self.config_cache[source].copy()
                
            if not os.path.exists(source):
                raise FileNotFoundError(f"Configuration file not found: {source}")
                
            with open(source, 'r') as f:
                config = json.load(f)
            
            self.config_cache[source] = config
            return config
            
        else:
            raise ValueError(f"Unsupported config source type: {type(source)}.")

    @abstractmethod
    def create_from_dict(self, config: Dict[str, Any]) -> Any:
        """
        Абстрактный метод создания объекта из словаря.
        Must be implemented in child classes.
        """
        pass

    def build(self, source: Union[str, Dict[str, Any]]) -> Any:
        """
        Основной метод сборки объекта.
        
        Main assembly method. Handles both file paths and dictionaries.
        """
        config = self.load_config(source)
        return self.create_from_dict(config)


class NetworkFactory(BaseFactory):
    """
    Фабрика для создания нейронных сетей Izhikevich.
    
    Factory for creating Izhikevich neural networks.
    """

    def create_from_dict(self, config: Dict[str, Any]) -> Izhikevich_IO_Network:
        """
        Создает сеть из словаря конфигурации.
        
        Creates network from configuration dictionary.
        
        Args:
            config: Словарь с параметрами сети.
            
        Returns:
            Izhikevich_IO_Network: Готовая сеть.
        """
        net_params = config['network_params']
        N = net_params['N']
        
        # 1. Параметры нейронов
        types = net_params.get('types', ['RS'] * N)
        A, B, C, D = types2params(types)
        names = net_params.get('names', [f"N_{i}" for i in range(N)])
        
        # 2. Загрузка матриц M, W и tau_syn напрямую из конфига
        weights_cfg = config.get('weights', {})
        
        if 'mask' not in weights_cfg or 'matrix' not in weights_cfg:
            raise ValueError("Config must contain both 'mask' and 'matrix' in 'weights' section.")
            
        M = np.array(weights_cfg['mask'])
        W = np.array(weights_cfg['matrix'])
        
        if M.shape != (N, N) or W.shape != (N, N):
            raise ValueError(f"Mask and Weights matrices must have shape ({N}, {N})")

        # 3. Синаптическая динамика (tau_syn тоже матрица N*N)
        synaptic_cfg = config.get('synaptic_dynamics', {})
        
        if 'tau_syn_ms' in synaptic_cfg:
            tau_syn_matrix = np.array(synaptic_cfg['tau_syn_ms'])
            if tau_syn_matrix.shape != (N, N):
                raise ValueError(f"Tau_syn matrix must have shape ({N}, {N})")
        else:
            # Если матрицы нет, используем скалярное значение по умолчанию для всех связей
            default_tau = synaptic_cfg.get('default_tau_ms', 20.0)
            tau_syn_matrix = default_tau * np.ones((N, N))
            
        # 4. Интерфейсы
        interfaces = config['interfaces']
        Q_app = np.array(interfaces['Q_app'])
        
        q_aff_cfg = interfaces.get('Q_aff', 'zeros')
        if isinstance(q_aff_cfg, str) and q_aff_cfg == 'zeros':
            Q_aff = np.zeros((N, net_params['afferent_size']))
        else:
            Q_aff = np.array(q_aff_cfg)
            
        P = np.array(interfaces['P'])
        
        # 5. Создание экземпляра сети
        # Передаем M, W и tau_syn_matrix напрямую в конструктор
        net = Izhikevich_IO_Network(
            N=N,
            input_size=net_params['input_size'],
            output_size=net_params['output_size'],
            afferent_size=net_params['afferent_size'],
            names=names,
            Q_app=Q_app,
            Q_aff=Q_aff,
            P=P,
            M=M,
            W=W,
            tau_syn=tau_syn_matrix
        )
        
        net.set_params(a=A, b=B, c=C, d=D)
        
        return net

class MuscleFactory(BaseFactory):
    """
    Фабрика для создания моделей мышц SimpleAdaptedMuscle.
    
    Factory for creating SimpleAdaptedMuscle models.
    """

    @staticmethod
    def create_from_dict(params: Dict[str, Any]) -> SimpleAdaptedMuscle:
        """
        Создает мышцу из словаря параметров.
        
        Creates a muscle from a dictionary of parameters.
        
        Args:
            params: Словарь с параметрами мышцы.
                Поддерживаемые ключи:
                - w: Вес синапса нейрон-мышца (default 0.5)
                - N: Количество саркомеров/единиц (default 50)
                - A: Коэффициент силы (default 0.0074)
                - tau_c: величина tau_c для сгибателя (default 39)
                - tau_1: величина tau_1 для сгибателя (default 21)
                
        Returns:
            SimpleAdaptedMuscle: Инициализированный объект мышцы.
        """
        # Извлечение параметров с значениями по умолчанию, соответствующими Var_Limb/Rybak2002
        w = params.get('w', 0.5)
        N = params.get('N', 50)
        A = params.get('A', 0.0074)
        
        # В наших конфигах мы часто храним обратные величины (частоты) для удобства
        tau_c = params.get('tau_c', params.get('tau_c', 39))
        tau_1 = params.get('tau_1', params.get('tau_1', 21))
        
        muscle = SimpleAdaptedMuscle(
            w=w,
            N=N,
            A=A,
            tau_c=tau_c,   # Конструктор ожидает частоту (1/ms), если передаем inv
            tau_1=tau_1
        )
        
        return muscle

    @staticmethod
    def create_standard_flexor() -> SimpleAdaptedMuscle:
        """
        Создает стандартный сгибатель с параметрами из статьи Markin et al. / Rybak.
        
        Creates a standard flexor with parameters from Markin et al. / Rybak.
        
        Returns:
            SimpleAdaptedMuscle: Готовый сгибатель.
        """
        return MuscleFactory.create_from_dict({
            'w': 0.5,
            'N': 50,
            'tau_c': 39,
            'tau_1': 21
        })

    @staticmethod
    def create_standard_extensor() -> SimpleAdaptedMuscle:
        """
        Создает стандартный разгибатель.
        
        Creates a standard extensor.
        
        Returns:
            SimpleAdaptedMuscle: Готовый разгибатель.
        """
        return MuscleFactory.create_from_dict({
            'w': 0.5,
            'N': 50,
            'tau_c': 39,
            'tau_1': 21
        })

class LimbFactory(BaseFactory):
    """
    Фабрика для создания механических моделей конечностей (OneDOFLimb).
    
    Factory for creating mechanical limb models (OneDOFLimb).
    """

    @staticmethod
    def create_from_dict(params: Dict[str, Any], with_gr: bool = False) -> OneDOFLimb:
        """
        Создает объект конечности из словаря параметров.
        
        Creates a limb object from a dictionary of parameters.
        
        Args:
            params: Словарь с параметрами конечности.
                Поддерживаемые ключи:
                - m: Масса сегмента (кг) (default 0.3)
                - ls: Длина сегмента (м) (default 0.3)
                - b: Коэффициент вязкого трения (default 0.01)
                - q0: Начальный угол (рад) (default pi/2)
                - w0: Начальная угловая скорость (рад/с) (default 0.0)
                - a1: Точка крепления мышцы 1 (м) (default 0.06)
                - a2: Точка крепления мышцы 2 (м) (default 0.007)
            with_gr: Если True, создает экземпляр OneDOFLimb_withGR (с реакцией опоры).
            
        Returns:
            OneDOFLimb: Инициализированный объект конечности.
        """
        # Извлечение параметров с значениями по умолчанию, соответствующими Var_Limb/Rybak2002
        limb_params = {
            'm': params.get('mass', 0.3),
            'ls': params.get('length', 0.3),
            'b': params.get('viscosity', 0.01),
            'q0': params.get('q0', np.pi / 2),
            'w0': params.get('w0', 0.0),
            'a1': params.get('tendon_a1', 0.06),
            'a2': params.get('tendon_a2', 0.007)
        }
        
        if with_gr:
            return OneDOFLimb_withGR(**limb_params)
        else:
            return OneDOFLimb(**limb_params)

    @staticmethod
    def create_standard_limb() -> OneDOFLimb:
        """
        Создает стандартную конечность с параметрами из статьи Markin et al. / Rybak.
        
        Creates a standard limb with parameters from Markin et al. / Rybak.
        
        Returns:
            OneDOFLimb: Готовая конечность.
        """
        return LimbFactory.create_from_dict({
            'mass': 0.3,
            'length': 0.3,
            'viscosity': 0.01,
            'q0': np.pi / 2,
            'tendon_a1': 0.06,
            'tendon_a2': 0.007
        })


class AfferentedLimbFactory(BaseFactory):
    """
    Фабрика для сборки афферентированных конечностей (Afferented_Limb).
    
    Factory for assembling afferented limbs (Afferented_Limb).
    """

    @staticmethod
    def create_from_dict(limb_cfg: Dict[str, Any]) -> Afferented_Limb:
        """
        Создает объект Afferented_Limb из словаря параметров.
        
        Creates an Afferented_Limb object from a dictionary of parameters.
        
        Args:
            limb_cfg: Словарь с параметрами конечности.
            
        Returns:
            Afferented_Limb: Готовая конечность.
            
        Raises:
            ValueError: Если конфигурация неполна или содержит ошибки.
        """
        try:
            # 1. Создаем механику
            mech_params = limb_cfg.get('mechanics', {})
            if not mech_params and 'mass' in limb_cfg:
                 mech_params = limb_cfg # Fallback
            
            try:
                limb_mech = LimbFactory.create_from_dict(mech_params)
            except KeyError as e:
                raise ValueError(f"Invalid mechanics configuration in limb config: missing key {e}")
            except Exception as e:
                raise ValueError(f"Failed to create limb mechanics: {str(e)}")
            
            # 2. Создаем мышцы
            flexor_cfg = limb_cfg.get('flexor', {})
            extensor_cfg = limb_cfg.get('extensor', {})
            
            try:
                flexor = MuscleFactory.create_from_dict(flexor_cfg)
            except KeyError as e:
                raise ValueError(f"Invalid flexor muscle configuration: missing key {e}")
                
            try:
                extensor = MuscleFactory.create_from_dict(extensor_cfg)
            except KeyError as e:
                raise ValueError(f"Invalid extensor muscle configuration: missing key {e}")
                
            # 3. Собираем конечность
            # Здесь мы передаем уже проверенные объекты, так что конструктор Afferented_Limb
            # должен отработать без ошибок, если объекты не None.
            if limb_mech is None or flexor is None or extensor is None:
                raise ValueError("One of the limb components (mechanics, flexor, extensor) is None.")
                
            return Afferented_Limb(
                Limb=limb_mech,
                Flexor=flexor,
                Extensor=extensor
            )
            
        except ValueError:
            # Пробрасываем наши кастомные ошибки дальше
            raise
        except Exception as e:
            # Ловим любые другие непредвиденные ошибки
            raise ValueError(f"Unexpected error during AfferentedLimb assembly: {str(e)}")

    @staticmethod
    def create_standard_limb() -> Afferented_Limb:
        return AfferentedLimbFactory.create_from_dict({})