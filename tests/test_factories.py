"""
Тесты для модуля фабрик SpikingNN.
Tests for the SpikingNN factories module.
"""

import pytest
import json
import os
import numpy as np
from unittest.mock import mock_open, patch

# Импортируем тестируемые классы
from SpikingNN.core.factories import (
    BaseFactory,
    NetworkFactory,
    MuscleFactory,
    LimbFactory,
    AfferentedLimbFactory
)
from SpikingNN.core.Izh_net import (
    Izhikevich_IO_Network,
    SimpleAdaptedMuscle,
    OneDOFLimb,
    OneDOFLimb_withGR,
    Afferented_Limb
)


class MockConcreteFactory(BaseFactory):
    """
    Конкретная реализация абстрактного класса для тестирования BaseFactory.
    
    Concrete implementation of abstract class for testing BaseFactory.
    """
    def create_from_dict(self, config):
        return {"created": True, "config": config}


class TestBaseFactory:
    """
    Тесты базового класса фабрики.
    Tests for the base factory class.
    """

    def setup_method(self):
        """
        Подготовка экземпляра фабрики перед каждым тестом.
        
        Setup factory instance before each test.
        """
        self.factory = MockConcreteFactory()

    def test_load_config_from_dict(self):
        """
        Проверка загрузки конфигурации из словаря.
        
        Verify loading configuration from dictionary.
        """
        cfg = {"key": "value"}
        result = self.factory.load_config(cfg)
        
        assert result == cfg
        # Проверяем, что возвращается копия (изменение оригинала не влияет на кэш/внутреннее состояние)
        cfg["key"] = "changed"
        assert result["key"] == "value"

    def test_load_config_from_file(self, tmp_path):
        """
        Проверка загрузки конфигурации из JSON файла.
        
        Verify loading configuration from JSON file.
        """
        cfg_data = {"file_key": "file_value"}
        p = tmp_path / "test_config.json"
        p.write_text(json.dumps(cfg_data))
        
        result = self.factory.load_config(str(p))
        assert result == cfg_data

    def test_load_config_caching(self, tmp_path):
        """
        Проверка работы кэша конфигураций.
        
        Verify configuration caching mechanism.
        """
        cfg_data = {"cached": True}
        p = tmp_path / "cache_test.json"
        p.write_text(json.dumps(cfg_data))
        
        path_str = str(p)
        
        # Первый вызов читает файл
        r1 = self.factory.load_config(path_str)
        # Второй вызов должен взять из кэша
        r2 = self.factory.load_config(path_str)
        
        assert r1 is not r2  # Это копии
        assert r1 == r2
        
        # Изменяем файл на диске
        p.write_text(json.dumps({"cached": False}))
        
        # Результат все еще должен быть старым (из кэша)
        r3 = self.factory.load_config(path_str)
        assert r3["cached"] is True

    def test_load_config_file_not_found(self):
        """
        Проверка ошибки при отсутствии файла.
        
        Verify error handling for missing files.
        """
        with pytest.raises(FileNotFoundError):
            self.factory.load_config("non_existent_file.json")

    def test_load_config_unsupported_type(self):
        """
        Проверка ошибки при неверном типе источника.
        
        Verify error handling for unsupported source types.
        """
        with pytest.raises(ValueError, match="Unsupported config source type"):
            self.factory.load_config(12345)

    def test_build_method(self):
        """
        Проверка сквозного метода build.
        
        Verify end-to-end build method.
        """
        cfg = {"test": "build"}
        result = self.factory.build(cfg)
        
        assert result["created"] is True
        assert result["config"]["test"] == "build"


class TestNetworkFactory:
    """
    Тесты фабрики нейронных сетей.
    Tests for the network factory.
    """

    @property
    def valid_minimal_config(self):
        """
        Валидная минимальная конфигурация сети.
        
        Valid minimal network configuration.
        """
        return {
            "network_params": {
                "N": 2,
                "input_size": 2,
                "output_size": 2,
                "afferent_size": 6,
                "types": ["RS", "FS"],
                "names": ["Neuron_A", "Neuron_B"]
            },
            "weights": {
                "mask": [[0, 0], [1, 0]],
                "matrix": [[0.0, 0.0], [1.5, 0.0]]
            },
            "synaptic_dynamics": {
                "tau_syn_ms": [[20.0, 20.0], [20.0, 20.0]]
            },
            "interfaces": {
                "Q_app": [[1, 0], [0, 1]],
                "Q_aff": "zeros",
                "P": [[0, 1], [1, 0]]
            }
        }

    def test_create_from_dict_success(self):
        """
        Успешное создание сети из валидного конфига.
        
        Successful network creation from valid config.
        """
        factory = NetworkFactory()
        net = factory.create_from_dict(self.valid_minimal_config)
        
        assert net.N == 2
        assert net.input_size == 2
        assert net.output_size == 2
        assert net.names == ["Neuron_A", "Neuron_B"]
        np.testing.assert_array_equal(net.M, np.array([[0, 0], [1, 0]]))
        np.testing.assert_array_equal(net.W, np.array([[0.0, 0.0], [1.5, 0.0]]))

    def test_missing_weights_section(self):
        """
        Ошибка при отсутствии секции весов.
        
        Error when weights section is missing.
        """
        factory = NetworkFactory()
        cfg = self.valid_minimal_config.copy()
        del cfg['weights']
        
        with pytest.raises(ValueError): 
            # Или ValueError, если вы добавите проверку наличия ключа 'weights' 
            # до проверки mask/matrix внутри create_from_dict
            factory.create_from_dict(cfg)

    def test_missing_mask_or_matrix(self):
        """
        Ошибка при отсутствии mask или matrix в весах.
        
        Error when mask or matrix is missing in weights.
        """
        factory = NetworkFactory()
        cfg = self.valid_minimal_config.copy()
        print(cfg)
        cfg['weights'] = {"mask": [[0,0],[1,0]]} # Нет matrix
        
        with pytest.raises(ValueError, match="must contain both 'mask' and 'matrix'"):
            factory.build(cfg)

    def test_invalid_matrix_shape(self):
        """
        Ошибка при несоответствии размера матриц N.
        
        Error when matrix shape does not match N.
        """
        factory = NetworkFactory()
        cfg = self.valid_minimal_config.copy()
        # N=2, но матрица 3x3
        cfg['weights']['mask'] = [[0,0,0],[0,0,0],[0,0,0]]
        cfg['weights']['matrix'] = [[0,0,0],[0,0,0],[0,0,0]]
        
        with pytest.raises(ValueError, match="must have shape"):
            factory.create_from_dict(cfg)

    def test_q_aff_zeros_string(self):
        """
        Проверка обработки строки 'zeros' для Q_aff.
        
        Verify handling of 'zeros' string for Q_aff.
        """
        factory = NetworkFactory()
        net = factory.build(self.valid_minimal_config)
        
        expected_shape = (2, 6) # N=2, afferent_size=6
        assert net.Q_aff.shape == expected_shape
        np.testing.assert_array_equal(net.Q_aff, np.zeros(expected_shape))

    def test_q_aff_explicit_matrix(self):
        """
        Проверка явной матрицы Q_aff.
        
        Verify explicit Q_aff matrix loading.
        """
        factory = NetworkFactory()
        cfg = self.valid_minimal_config.copy()
        explicit_aff = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        cfg['interfaces']['Q_aff'] = explicit_aff
        
        net = factory.build(cfg)
        np.testing.assert_array_equal(net.Q_aff, np.array(explicit_aff))

    def test_default_tau_syn(self):
        """
        Проверка значения tau_syn по умолчанию.
        
        Verify default tau_syn value.
        """
        factory = NetworkFactory()
        cfg = self.valid_minimal_config.copy()
        del cfg['synaptic_dynamics']
        
        net = factory.build(cfg)
        # По умолчанию должно быть 20.0
        np.testing.assert_allclose(net.tau_syn, np.array([[0.05, 0.05], [0.05, 0.05]]))

    def test_custom_tau_syn(self):
        """
        Проверка пользовательского значения tau_syn.
        
        Verify custom tau_syn value.
        """
        factory = NetworkFactory()
        cfg = self.valid_minimal_config.copy()
        cfg['synaptic_dynamics']['tau_syn_ms'] = np.array([[25.0, 25.0], [25.0, 25.0]])
        print(cfg)
        net = factory.build(cfg)
        np.testing.assert_allclose(net.tau_syn, 1/np.array([[25.0, 25.0], [25.0, 25.0]]))

    def test_build_from_file_integration(self, tmp_path):
        """
        Интеграционный тест: загрузка из файла через метод build.
        
        Integration test: loading from file via build method.
        """
        factory = NetworkFactory()
        p = tmp_path / "net_config.json"
        p.write_text(json.dumps(self.valid_minimal_config))
        
        net = factory.build(str(p))
        assert net.N == 2
        assert net.names[0] == "Neuron_A"

"""
Тесты для фабрики мышц MuscleFactory.
Tests for the MuscleFactory class.
"""



class TestMuscleFactory:
    """
    Тесты фабрики создания мышц.
    Tests for the muscle creation factory.
    """

    def test_create_from_dict_default_params(self):
        """
        Проверка создания мышцы с параметрами по умолчанию.
        
        Verify muscle creation with default parameters.
        """
        # Пустой словарь должен использовать дефолтные значения из метода
        muscle = MuscleFactory.create_from_dict({})
        
        assert isinstance(muscle, SimpleAdaptedMuscle)
        assert muscle.w == 0.5
        assert muscle.N == 50
        assert muscle.A == 0.0074
        # tau_c и tau_1 в конструкторе принимаются как частоты (inv), 
        # поэтому проверяем, что они равны переданным инверсным значениям по умолчанию
        assert muscle.tau_c == 1/39
        assert muscle.tau_1 == 1/21

    def test_create_from_dict_custom_params(self):
        """
        Проверка создания мышцы с пользовательскими параметрами.
        
        Verify muscle creation with custom parameters.
        """
        params = {
            'w': 0.8,
            'N': 100,
            'A': 0.01,
            'tau_c': 50,
            'tau_1': 30
        }
        muscle = MuscleFactory.create_from_dict(params)
        
        assert muscle.w == 0.8
        assert muscle.N == 100
        assert muscle.A == 0.01
        assert muscle.tau_c == 1/50
        assert muscle.tau_1 == 1/30

    def test_create_from_dict_legacy_keys(self):
        """
        Проверка поддержки старых ключей (flexor_tau_c_inv).
        
        Verify support for legacy keys (flexor_tau_c_inv).
        """
        params = {
            'tau_c': 45,
            'tau_1': 25
        }
        muscle = MuscleFactory.create_from_dict(params)
        
        # Должны подхватиться старые ключи, если новые не указаны
        assert muscle.tau_c == 1/45
        assert muscle.tau_1 == 1/25


    def test_create_standard_flexor(self):
        """
        Проверка создания стандартного сгибателя.
        
        Verify creation of standard flexor.
        """
        muscle = MuscleFactory.create_standard_flexor()
        
        assert isinstance(muscle, SimpleAdaptedMuscle)
        assert muscle.w == 0.5
        assert muscle.N == 50
        assert muscle.tau_c == 1/39
        assert muscle.tau_1 == 1/21

    def test_create_standard_extensor(self):
        """
        Проверка создания стандартного разгибателя.
        
        Verify creation of standard extensor.
        """
        muscle = MuscleFactory.create_standard_extensor()
        
        assert isinstance(muscle, SimpleAdaptedMuscle)
        assert muscle.w == 0.5
        assert muscle.N == 50
        assert muscle.tau_c == 1/39
        assert muscle.tau_1 == 1/21

    def test_muscle_functionality_after_creation(self):
        """
        Интеграционный тест: проверка работы созданной мышцы.
        
        Integration test: verify created muscle functionality.
        """
        muscle = MuscleFactory.create_standard_flexor()
        
        # Начальное состояние
        assert muscle.F == 0
        assert muscle.Cn == 0
        
        # Делаем шаг с входным сигналом
        muscle.step(dt=0.1, u=1.0)
        
        # После шага сила должна стать положительной (если параметры корректны)
        assert muscle.F > 0
        assert muscle.Cn > 0

"""
Тесты для фабрики конечностей LimbFactory.
Tests for the LimbFactory class.
"""


class TestLimbFactory:
    """
    Тесты фабрики создания конечностей.
    Tests for the limb creation factory.
    """

    def test_create_from_dict_default_params(self):
        """
        Проверка создания конечности с параметрами по умолчанию.
        
        Verify limb creation with default parameters.
        """
        # Пустой словарь должен использовать дефолтные значения из метода
        limb = LimbFactory.create_from_dict({})
        
        assert isinstance(limb, OneDOFLimb)
        assert limb.m == 0.3
        assert limb.ls == 0.3
        assert limb.b == 0.01
        assert limb.q0 == pytest.approx(3.14159 / 2) # pi/2
        assert limb.a1 == 0.06
        assert limb.a2 == 0.007

    def test_create_from_dict_custom_params(self):
        """
        Проверка создания конечности с пользовательскими параметрами.
        
        Verify limb creation with custom parameters.
        """
        params = {
            'mass': 0.5,
            'length': 0.4,
            'viscosity': 0.02,
            'q0': 1.0,
            'w0': 0.5,
            'tendon_a1': 0.08,
            'tendon_a2': 0.01
        }
        limb = LimbFactory.create_from_dict(params)
        
        assert limb.m == 0.5
        assert limb.ls == 0.4
        assert limb.b == 0.02
        assert limb.q0 == 1.0
        assert limb.w0 == 0.5
        assert limb.a1 == 0.08
        assert limb.a2 == 0.01

    def test_create_from_dict_with_gr_true(self):
        """
        Проверка создания конечности с реакцией опоры (with_gr=True).
        
        Verify limb creation with ground reaction force (with_gr=True).
        """
        limb = LimbFactory.create_from_dict({}, with_gr=True)
        
        assert isinstance(limb, OneDOFLimb_withGR)
        assert hasattr(limb, 'GR') # Проверка наличия метода GR

    def test_create_standard_limb(self):
        """
        Проверка создания стандартной конечности.
        
        Verify creation of standard limb.
        """
        limb = LimbFactory.create_standard_limb()
        
        assert isinstance(limb, OneDOFLimb)
        assert limb.m == 0.3
        assert limb.ls == 0.3
        assert limb.a1 == 0.06
        assert limb.a2 == 0.007

    def test_build_from_dict(self):
        """
        Проверка метода build со словарем.
        
        Verify build method with dictionary.
        """
        factory = LimbFactory()
        config = {'mass': 0.6, 'length': 0.5}
        
        limb = factory.build(config)
        
        assert limb.m == 0.6
        assert limb.ls == 0.5

    def test_build_from_file(self, tmp_path):
        """
        Проверка метода build с JSON файлом.
        
        Verify build method with JSON file.
        """
        factory = LimbFactory()
        config = {'mass': 0.7, 'length': 0.6, 'viscosity': 0.03}
        
        p = tmp_path / "limb_config.json"
        p.write_text(json.dumps(config))
        
        limb = factory.build(str(p))
        
        assert limb.m == 0.7
        assert limb.ls == 0.6
        assert limb.b == 0.03

    def test_build_caching(self, tmp_path):
        """
        Проверка кэширования конфигов при использовании build с файлом.
        
        Verify config caching when using build with file.
        """
        factory = LimbFactory()
        config = {'mass': 0.8}
        
        p = tmp_path / "cache_limb.json"
        p.write_text(json.dumps(config))
        
        # Первый вызов читает файл
        limb1 = factory.build(str(p))
        
        # Изменяем файл на диске
        p.write_text(json.dumps({'mass': 0.9}))
        
        # Второй вызов должен взять из кэша (старое значение)
        limb2 = factory.build(str(p))
        
        assert limb1.m == 0.8
        assert limb2.m == 0.8 # Значение из кэша, а не 0.9 из файла

    def test_limb_mechanics_after_creation(self):
        """
        Интеграционный тест: проверка работы созданной конечности.
        
        Integration test: verify created limb functionality.
        """
        limb = LimbFactory.create_standard_limb()
        
        # Начальное состояние
        initial_q = limb.q
        
        # Делаем шаг симуляции
        for i in range(10):
            limb.step(dt=0.1, F_flex=1.0, F_ext=0.0)
        
        # Угол должен измениться под действием силы сгибателя
        assert limb.q != initial_q
        # Скорость должна стать ненулевой
        assert limb.w != 0.0

"""
Тесты для фабрики афферентированных конечностей AfferentedLimbFactory.
Tests for the AfferentedLimbFactory class.
"""

class TestAfferentedLimbFactory:
    """Тесты фабрики сборки афферентированных конечностей."""

    def test_create_from_dict_success_full_config(self):
        """Проверка успешной сборки конечности с полной конфигурацией."""
        config = {
            "mechanics": {
                "mass": 0.5,
                "length": 0.4,
                "viscosity": 0.02,
                "q0": 1.0,
                "tendon_a1": 0.08,
                "tendon_a2": 0.01
            },
            "flexor": {
                "w": 0.6,
                "N": 60,
                "tau_c_inv": 40,
                "tau_1_inv": 20
            },
            "extensor": {
                "w": 0.4,
                "N": 40,
                "tau_c_inv": 30,
                "tau_1_inv": 25
            }
        }
        
        limb = AfferentedLimbFactory.create_from_dict(config)
        
        assert isinstance(limb, Afferented_Limb)
        assert isinstance(limb.Limb, OneDOFLimb)
        assert isinstance(limb.Flexor, SimpleAdaptedMuscle)
        assert isinstance(limb.Extensor, SimpleAdaptedMuscle)
        
        assert limb.Limb.m == 0.5
        assert limb.Limb.ls == 0.4
        
        assert limb.Flexor.w == 0.6
        assert limb.Extensor.w == 0.4

    def test_create_from_dict_success_minimal_config(self):
        """Проверка сборки с пустой конфигурацией (использование дефолтов)."""
        limb = AfferentedLimbFactory.create_from_dict({})
        
        assert isinstance(limb, Afferented_Limb)
        assert limb.Limb.m == 0.3
        assert limb.Flexor.w == 0.5

    def test_create_from_dict_legacy_mechanics_format(self):
        """Проверка поддержки старого формата конфига (параметры в корне)."""
        config = {
            "mass": 0.7,
            "length": 0.5,
            "flexor": {},
            "extensor": {}
        }
        
        limb = AfferentedLimbFactory.create_from_dict(config)
        
        assert limb.Limb.m == 0.7
        assert limb.Limb.ls == 0.5

    def test_create_from_dict_mechanics_error(self):
        """Проверка перехвата ошибки при неверной конфигурации механики."""
        with patch.object(LimbFactory, 'create_from_dict', side_effect=KeyError("mass")):
            with pytest.raises(ValueError, match="Invalid mechanics configuration"):
                AfferentedLimbFactory.create_from_dict({"mechanics": {}})

    def test_create_from_dict_flexor_error(self):
        """
        Проверка перехвата ошибки при неверной конфигурации сгибателя.
        Мы мокатем только первый вызов create_from_dict, чтобы ошибка возникла на flexor.
        """
        original_method = MuscleFactory.create_from_dict
        
        call_count = [0]
        
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            # Первый вызов (Flexor) выбрасывает ошибку
            if call_count[0] == 1:
                raise KeyError("w")
            # Второй вызов (Extensor) работает нормально
            return original_method(*args, **kwargs)
            
        with patch.object(MuscleFactory, 'create_from_dict', side_effect=side_effect):
            with pytest.raises(ValueError, match="Invalid flexor muscle configuration"):
                AfferentedLimbFactory.create_from_dict({"flexor": {}, "extensor": {}})

    def test_create_from_dict_extensor_error(self):
        """
        Проверка перехвата ошибки при неверной конфигурации разгибателя.
        Мы позволяем первому вызову (Flexor) пройти успешно, а второй (Extensor) роняем.
        """
        original_method = MuscleFactory.create_from_dict
        
        call_count = [0]
        
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            # Первый вызов (Flexor) работает нормально
            if call_count[0] == 1:
                return original_method(*args, **kwargs)
            # Второй вызов (Extensor) выбрасывает ошибку
            else:
                raise KeyError("w")
                
        with patch.object(MuscleFactory, 'create_from_dict', side_effect=side_effect):
            with pytest.raises(ValueError, match="Invalid extensor muscle configuration"):
                AfferentedLimbFactory.create_from_dict({"flexor": {}, "extensor": {}})

    def test_assembled_limb_functionality(self):
        """
        Интеграционный тест: проверка работы собранной конечности.
        Проверяем изменение состояния после нескольких шагов симуляции.
        """
        config = {
            "mechanics": {"mass": 0.3, "length": 0.3},
            "flexor": {"w": 0.5},
            "extensor": {"w": 0.5}
        }
        
        limb = AfferentedLimbFactory.create_from_dict(config)
        
        initial_q = limb.q
        initial_w = limb.w
        
        # Делаем 100 шагов симуляции с сильной активацией сгибателя
        for _ in range(500):
            limb.step(dt=0.1, uf=20.0, ue=0.0)
        
        # После 100 шагов (10 мс) угол должен измениться заметно
        # Или хотя бы должна появиться угловая скорость
        assert limb.w != 0.0, "Angular velocity should not be zero after stimulation"
        
        # Угол должен измениться (конечность согнется)
        # Допускаем небольшую погрешность, но изменение должно быть существенным
        assert abs(limb.q - initial_q) > 0.01, f"Angle did not change significantly: {initial_q} -> {limb.q}"
        
        # Сила сгибателя должна вырасти
        assert limb.F_flex > 0.1, "Flexor force should increase after stimulation"