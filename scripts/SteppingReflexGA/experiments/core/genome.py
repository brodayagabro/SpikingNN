"""Bounded, normalized genome for the two-neuron circuit in the notebook."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from .config import MECHANICS, DT_MS, MODEL


@dataclass(frozen=True)
class Gene:
    name: str
    low: float
    high: float
    initial: float
    log: bool = False


def _genes():
    genes = []
    for name, lo, hi, initial, log in (
        ('a', .005, .15, .1, True), ('b', .15, .3, .2, False),
        ('c', -70, -45, -65, False), ('d', .1, 8, 2, True),
    ):
        genes.extend(Gene(f'neurons.{name}.{i}', lo, hi, initial, log) for i in range(2))
    genes.extend(Gene(f'synapses.W.{i}.{j}', -2, 2, -.1 if i != j else 0)
                 for i in range(2) for j in range(2))
    genes.extend(Gene(f'synapses.tau_ms.{i}.{j}', 2, 60, 20, True)
                 for i in range(2) for j in range(2))
    for side in ('flexor', 'extensor'):
        for name, lo, hi, initial, log in (
            ('w', .02, 1.5, .15, True), ('N', 1, 40, 2, False),
            ('A', .001, .02, .0074, True), ('tau_c', 5, 100, 20, True),
            ('tau_1', 5, 100, 20, True), ('m', 1.5, 4, 2.5, False),
            ('k', .1, 2, .75, True),
        ):
            genes.append(Gene(f'muscles.{side}.{name}', lo, hi, initial, log))
    # Contralateral stretch excites the corrective actuator at q = -pi/2.
    initial_feedback = [[-40, -600, -1, 40, 600, 0], [40, 600, 0, -40, -600, -1]]
    for i in range(2):
        for j, kind in enumerate(('Ia_F', 'II_F', 'Ib_F', 'Ia_E', 'II_E', 'Ib_E')):
            bound = (200, 2000, 30)[j % 3]
            genes.append(Gene(f'feedback.Q_aff.{i}.{kind}', -bound, bound, initial_feedback[i][j]))
    length = float(np.hypot(.2, .05))
    genes.extend(Gene(f'feedback.L_th.{i}', .85 * length, 1.1 * length, length) for i in range(2))
    genes.extend(Gene(f'feedback.F_th.{i}', .1, 5, 3.38, True) for i in range(2))
    genes.extend(Gene(f'input.bias_nA.{i}', 0, 8, 0) for i in range(2))
    genes.extend(Gene(f'output.P.{i}.{j}', 0, 2, float(i == j)) for i in range(2) for j in range(2))
    return tuple(genes)


GENES = _genes()
SIZE = len(GENES)
LOW = np.array([np.log(g.low) if g.log else g.low for g in GENES])
HIGH = np.array([np.log(g.high) if g.log else g.high for g in GENES])
LOG = np.array([g.log for g in GENES])


def encode(physical):
    physical = np.asarray(physical, dtype=float)
    if physical.shape != (SIZE,) or not np.isfinite(physical).all():
        raise ValueError(f'Expected {SIZE} finite physical parameters')
    if np.any(physical[LOG] <= 0):
        raise ValueError('Logarithmic parameters must be positive')
    transformed = physical.copy()
    transformed[LOG] = np.log(transformed[LOG])
    vector = (transformed - LOW) / (HIGH - LOW)
    if np.any(vector < -1e-12) or np.any(vector > 1 + 1e-12):
        raise ValueError('Physical parameters outside genome bounds')
    return np.clip(vector, 0, 1)


def decode(vector):
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (SIZE,) or not np.isfinite(vector).all() or np.any((vector < 0) | (vector > 1)):
        raise ValueError(f'Expected {SIZE} finite genes in [0, 1]')
    physical = LOW + vector * (HIGH - LOW)
    physical[LOG] = np.exp(physical[LOG])
    return physical


def initial_genome():
    return encode([gene.initial for gene in GENES])


def parameters(vector):
    p = decode(vector)
    return {
        'neurons': {key: p[2*i:2*i+2].tolist() for i, key in enumerate(('a', 'b', 'c', 'd'))},
        'synapses': {'weights_target_source': p[8:12].reshape(2, 2).tolist(),
                     'tau_ms': p[12:16].reshape(2, 2).tolist()},
        'muscles': {side: dict(zip(('w', 'N', 'A', 'tau_c', 'tau_1', 'm', 'k'),
                                  p[16+7*i:23+7*i].tolist()))
                    for i, side in enumerate(('flexor', 'extensor'))},
        'feedback': {'Q_aff': p[30:42].reshape(2, 6).tolist(),
                     'channel_order': ['Ia_F', 'II_F', 'Ib_F', 'Ia_E', 'II_E', 'Ib_E'],
                     'L_th': p[42:44].tolist(), 'F_th': p[44:46].tolist()},
        'bias_nA': p[46:48].tolist(), 'P': p[48:52].reshape(2, 2).tolist(),
    }


def save_genome(path, vector, **metadata):
    physical = decode(vector)
    payload = {'version': 1, 'genes': [asdict(g) for g in GENES],
               'normalized': np.asarray(vector).tolist(), 'parameters': parameters(vector),
               'hyperparameters': {'comment': 'Фиксированные параметры; не входят в геном и не мутируют.',
                                   'mechanics': asdict(MECHANICS), 'dt_ms': DT_MS,
                                   'afferent_model': MODEL},
               'metadata': metadata}
    for index, (entry, value) in enumerate(zip(payload['genes'], physical)):
        description, parameter, unit = describe_gene(entry['name'])
        entry.update(index=index, value=float(value), description=description,
                     system_parameter=parameter, unit=unit)
    Path(path).write_text(format_json(payload) + '\n')


def describe_gene(name):
    """Human-readable comments stored as valid JSON fields, not JSON // syntax."""
    fields = name.split('.')
    group, key = fields[:2]
    if group == 'neurons':
        meanings = {'a': ('Скорость восстановления переменной U', '1/мс'),
                    'b': ('Чувствительность U к мембранному потенциалу V', 'ед. модели'),
                    'c': ('Потенциал сброса после спайка', 'мВ'),
                    'd': ('Приращение U при сбросе после спайка', 'ед. модели')}
        text, unit = meanings[key]
        return f'{text}; мотонейрон {fields[2]}.', f'Izhikevich_IO_Network.{key}[{fields[2]}]', unit
    if group == 'synapses':
        target, source = fields[2:]
        if key == 'W':
            return (f'Вес синапса от нейрона {source} к нейрону {target}; знак задаёт возбуждение/торможение.',
                    f'Izhikevich_IO_Network.W[{target},{source}]', 'ед. модели')
        return (f'Время затухания синаптического тока от нейрона {source} к нейрону {target}.',
                f'Izhikevich_IO_Network.tau_syn[{target},{source}] (вход конструктора)', 'мс')
    if group == 'muscles':
        muscle = 'сгибатель' if key == 'flexor' else 'разгибатель'
        parameter = fields[2]
        meanings = {'w': ('Вес нейромышечного синапса; множитель входа в динамике кальция', 'ед. модели'),
                    'N': ('Эффективное число мышечных единиц; множитель силы', 'безразмерная'),
                    'A': ('Масштаб генерации силы одной единицей', 'Н/мс'),
                    'tau_c': ('Время затухания кальция Cn', 'мс'),
                    'tau_1': ('Время релаксации силы F', 'мс'),
                    'm': ('Показатель степени в x=Cn^m/(Cn^m+k^m)', 'безразмерная'),
                    'k': ('Полуактивационная концентрация кальция', 'ед. Cn модели')}
        text, unit = meanings[parameter]
        return f'{text}; {muscle}.', f'{key}.SimpleAdaptedMuscle.{parameter} (вход конструктора)', unit
    if group == 'feedback':
        if key == 'Q_aff':
            neuron, channel = fields[2:]
            return (f'Вес афферентного канала {channel} на мотонейрон {neuron}; F — сгибатель, E — разгибатель.',
                    f'Izhikevich_IO_Network.Q_aff[{neuron},{channel}]', 'ед. модели')
        muscle = 'сгибатель' if fields[2] == '0' else 'разгибатель'
        text, unit = ('Пороговая длина растяжения для Ia и II', 'м') if key == 'L_th' else ('Пороговая сила для Ib', 'Н')
        return f'{text}; {muscle}.', f'Afferents[{fields[2]}].{key}', unit
    if group == 'input':
        return (f'Постоянный внешний ток на мотонейрон {fields[2]}; в опытах равновесия фиксирован на нуле.',
                f'Izhikevich_IO_Network.step(Iapp)[{fields[2]}]', 'ед. тока модели')
    if group == 'output':
        muscle, neuron = fields[2:]
        return (f'Проекция спайкового выхода нейрона {neuron} на мышцу {muscle} (0 — сгибатель, 1 — разгибатель).',
                f'Izhikevich_IO_Network.P[{muscle},{neuron}]', 'безразмерная')
    raise ValueError(f'Unknown genome field: {name}')


def format_json(value, level=0):
    """Pretty JSON with each vector / matrix row on a single line."""
    indent = '  ' * level
    child_indent = indent + '  '
    if isinstance(value, dict) and value:
        return '{\n' + ',\n'.join(
            child_indent + json.dumps(key, ensure_ascii=False) + ': ' + format_json(item, level+1)
            for key, item in value.items()) + '\n' + indent + '}'
    if isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value):
        return '[\n' + ',\n'.join(child_indent + format_json(item, level+1) for item in value) + '\n' + indent + ']'
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def load_genome(path):
    payload = json.loads(Path(path).read_text())
    if payload.get('version') != 1 or [g['name'] for g in payload['genes']] != [g.name for g in GENES]:
        raise ValueError('Incompatible genome schema')
    vector = np.array(payload['normalized'], dtype=float)
    decode(vector)
    return vector


# The canonical 52-field representation remains readable for historical data.
# New balance searches have only 50 free genes: applied currents are exactly zero.
WEIGHT_INDICES = np.array([16, 23])
BIAS_INDICES = np.array([46, 47])
FREE_INDICES = np.setdiff1d(np.arange(SIZE), BIAS_INDICES)


@dataclass(frozen=True)
class ControllerDomain:
    small_weights: bool = False

    @property
    def size(self):
        return len(FREE_INDICES)

    def _high(self):
        high = np.ones(SIZE)
        if self.small_weights:
            upper = decode(high)
            upper[WEIGHT_INDICES] = np.nextafter(.1, 0.)
            high = encode(upper)
            # A strict bound, including exp/log roundoff at the endpoint.
            while np.any(decode(high)[WEIGHT_INDICES] >= .1):
                high[WEIGHT_INDICES] = np.nextafter(high[WEIGHT_INDICES], 0.)
        return high

    def decode(self, unit):
        unit = np.asarray(unit, dtype=float)
        if unit.shape != (self.size,) or not np.isfinite(unit).all() or np.any((unit < 0) | (unit > 1)):
            raise ValueError("Expected 50 finite normalized controller genes")
        canonical = np.zeros(SIZE)
        canonical[FREE_INDICES] = unit * self._high()[FREE_INDICES]
        return decode(canonical)

    def encode(self, physical):
        physical = np.asarray(physical, dtype=float)
        if physical.shape != (SIZE,) or np.any(physical[BIAS_INDICES] != 0):
            raise ValueError("Balance controllers require exactly zero applied currents")
        unit = encode(physical)[FREE_INDICES] / self._high()[FREE_INDICES]
        if np.any(unit > 1 + 1e-12) or (self.small_weights and np.any(physical[WEIGHT_INDICES] >= .1)):
            raise ValueError("Neuromuscular weights outside the search domain")
        return np.clip(unit, 0, 1)

    def initial(self):
        physical = decode(initial_genome())
        if self.small_weights:
            physical[WEIGHT_INDICES] = .08
        return self.encode(physical)
