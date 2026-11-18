#!/usr/bin/env python3
"""CPG с афферентной обратной связью и конечностью с одной степенью свободы.

Быстрый запуск из SpikingNN/scripts/CPG_afferents_control:
    conda activate CPG
    python simulate.py
    python simulate.py --duration 20 --output dynamics_20s.png
    python simulate.py --genome genome.json --duration 10
    python simulate.py --no-feedback --output dynamics_no_feedback.png
    python simulate.py --genome runs/pilot_sweep/best_genome.json --drive 1.2

Нужны Python >= 3.10, numpy, matplotlib, jsonschema, tqdm. Скрипт подключает
SpikingNN из src/ текущего репозитория; устанавливать пакет отдельно не нужно.
Недостающие зависимости: python -m pip install numpy matplotlib jsonschema tqdm
Для переноса за пределы репозитория сначала установите этот checkout SpikingNN.

Три исходных файла:
  system.json — конфигурация, напрямую принимаемая SpikingNN.api.Simulation.
                Здесь фиксируются топология, механика, dt и длительность.
  genome.json — стартовые значения, границы, порядок и адреса изменяемых генов;
                также источник модели, обозначения и отличия от статьи.
  simulate.py — применение генома, запуск API, проверка чисел и одна PNG-фигура.
Значения генома перекрывают соответствующие значения system.json. Для правки
настраиваемых параметров меняйте genes[*].value в genome.json. Начальный JSON
системы и приложенный геном содержат одинаковые численные значения.

Источник: Markin et al. (2010), Afferent control of locomotor CPG: insights from
a simple neuromechanical model. DOI: 10.1111/j.1749-6632.2010.05435.x.
https://pmc.ncbi.nlm.nih.gov/articles/PMC5323435/
Перенесены 10 популяций, 14 связей между ними и 14 афферентных проекций
из Figure 1 / Table 1. Каждая популяция заменена ОДНИМ нейроном Ижикевича.
Это адаптация структуры, а не воспроизведение ионных уравнений/результатов
статьи. RG-F/E имеют замедленную адаптацию a=0.005. Веса исходной таблицы
масштабированы для токовых синапсов API; драйвы подобраны вручную. Различие
постоянных драйвов RG-F/E на 2% нарушает симметрию начальных условий.
Внешнего периодического сигнала нет. Геном ещё не проходил обучение.

Механика и мышцы — существующие OneDOFLimb и SimpleAdaptedMuscle SpikingNN.
Свободная конечность без контакта с опорой/ограничителей угла; q=+pi/2 —
нижняя вертикаль, q=-pi/2 — верхняя. Стартуем в нижней вертикали. Масса 0.3 kg,
длина 0.3 m, вязкость 0.001 N*m*s/rad, геометрия сухожилий a1=0.2, a2=0.05 m.
Мышцы не воспроизводят Hill-модель статьи; N=1, m=2.5, k=0.75 фиксированы.
Афференты вычисляет API: Ia (растяжение/скорость), II (растяжение), Ib (сила).
L_th=sqrt(a1^2+a2^2), F_th=3.38 N. В сети используются Ia-F, II-F, Ia-E, Ib-E;
Ib-F и II-E записываются, но не имеют проекций, как в схеме статьи.

Единицы: в system.json время и tau задаются в ms, q в rad, w в rad/s,
длины в m, силы в N. CLI --duration задаётся в СЕКУНДАХ. Веса и драйвы —
токовые коэффициенты API, а не проводимости из статьи. dt=0.1 ms фиксирует
также площадь спайкового импульса в текущем API; при смене dt потребуется
перекалибровка синаптических и нейромышечных весов.

Порядок вычислений API на каждом шаге:
  афференты прошлого шага -> сеть -> выходы Mn-F/E -> мышцы -> механика ->
  новые афференты. Истории записываются перед шагом; t лежит в [0, duration).
Матрицы W, tau_syn: [приёмник, источник]; Q_app/Q_aff: [нейрон, вход];
P: [мышца, нейрон]; connectivity: [источник, приёмник, знак]. Два входа
постоянны и равны 1: первый — нисходящий драйв, второй — драйв нейрона In.

Для будущего генетического алгоритма:
  g = json.loads(Path('genome.json').read_text())
  z = encode_genome(g)                 # нормированный вектор в [0, 1]
  child = decode_genome(z, g)          # после мутации/скрещивания z
  config = build_config(system, child)
  data = simulate(config)              # новый объект Simulation при каждом вызове
Топология, знаки связей, механика и соответствие выходов мышцам фиксированы.
Порядок genes — порядок хромосомы; его следует сохранять между поколениями.
Выход за диапазон отвергается, а не обрезается. Обучение слежению за синусоидой
реализовано отдельно в train.py; описание фитнесса и sweep находится в README.md.
--drive умножает только нисходящий вход d1 (RG и PF), оставляя d2 для In равным 1.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LOCAL_SRC = HERE.parent.parent / 'src'
if (LOCAL_SRC / 'SpikingNN' / 'api.py').is_file():
    sys.path.insert(0, str(LOCAL_SRC))

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
import numpy as np
from SpikingNN.api import Simulation, run_simulation
from SpikingNN.schema import validate_config

NEURONS = ('RG-F', 'RG-E', 'In-F', 'In-E', 'PF-F', 'PF-E', 'In', 'Inab-E', 'Mn-F', 'Mn-E')
AFFERENTS = ('Ia-F', 'II-F', 'Ib-F', 'Ia-E', 'II-E', 'Ib-E')
# Table 1, signed coefficients BEFORE adaptation to SpikingNN current synapses.
PAPER_CONNECTIONS = (
    (0, 2, .41), (0, 4, .7), (1, 3, .41), (1, 5, .7),
    (4, 8, 1.95), (5, 7, .35), (5, 9, 1.30), (7, 9, .82),
    (2, 1, -2.2), (2, 5, -6.6), (3, 0, -2.2), (3, 4, -6.6),
    (3, 6, -2.8), (6, 7, -.55),
)
# (afferent index, target neuron, Table 1 coefficient).
PAPER_AFFERENTS = (
    (0, 0, .06), (0, 2, .27), (0, 4, .19),
    (1, 0, .0348), (1, 2, .1566), (1, 4, .1102),
    (3, 1, .06), (3, 3, .44), (3, 5, .10), (3, 7, .16),
    (5, 1, .066), (5, 3, .484), (5, 5, .11), (5, 7, .176),
)
FLEXOR, EXTENSOR = '#2878a5', '#d6652a'


def _gene_records(genome):
    """Validate the chromosome description, preserving its explicit list order."""
    if genome.get('format') != 'cpg-afferents-genome-v1':
        raise ValueError('Неизвестный формат генома')
    genes = genome.get('genes')
    if not isinstance(genes, list) or not genes:
        raise ValueError('genes должен быть непустым списком')
    names, paths = set(), set()
    for gene in genes:
        name, path = gene['name'], tuple(gene['path'])
        if name in names or path in paths:
            raise ValueError(f'Повторяющийся ген или адрес: {name}')
        names.add(name)
        paths.add(path)
        low, high = gene['bounds']
        value = gene['value']
        if not np.isfinite([low, high, value]).all() or not low <= value <= high or low >= high:
            raise ValueError(f'Неверное значение/границы гена {name}: {value}, [{low}, {high}]')
        if gene['scale'] not in ('linear', 'log') or (gene['scale'] == 'log' and low <= 0):
            raise ValueError(f'Неверная шкала гена {name}')
    return genes


def encode_genome(genome):
    """Physical gene values -> flat [0,1] chromosome, with optional log scaling."""
    vector = []
    for gene in _gene_records(genome):
        low, high = gene['bounds']
        value = gene['value']
        if gene['scale'] == 'log':
            low, high, value = np.log([low, high, value])
        vector.append((value - low) / (high - low))
    return np.asarray(vector)


def decode_genome(vector, template):
    """Decode a bounded chromosome into a copy of a genome; never mutate template."""
    genes = _gene_records(template)
    z = np.asarray(vector, dtype=float)
    if z.shape != (len(genes),) or not np.isfinite(z).all() or np.any((z < 0) | (z > 1)):
        raise ValueError(f'Ожидается {len(genes)} конечных значений в [0, 1]')
    result = deepcopy(template)
    for x, gene in zip(z, result['genes']):
        low, high = gene['bounds']
        if x == 0 or x == 1:
            value = low if x == 0 else high
        elif gene['scale'] == 'log':
            value = np.exp(np.log(low) + x * (np.log(high) - np.log(low)))
        else:
            value = low + x * (high - low)
        gene['value'] = float(value)
    _gene_records(result)
    return result


def _allowed_paths():
    """Only existing connections and selected continuous parameters may evolve."""
    paths = {('network', 'neuron_params', key, i) for key in 'abcd' for i in range(10)}
    paths |= {('network', key, target, source) for source, target, _ in PAPER_CONNECTIONS
              for key in ('weights', 'tau_syn')}
    paths |= {('network', 'Q_aff', target, source) for source, target, _ in PAPER_AFFERENTS}
    paths |= {('network', 'Q_app', target, channel) for target, channel in ((0, 0), (1, 0), (4, 0), (5, 0), (6, 1))}
    paths |= {('limbs', 0, muscle, key) for muscle in ('flexor', 'extensor') for key in ('w', 'A', 'tau_c', 'tau_1')}
    return paths


def validate_system(config):
    """Validate the API schema and preserve the intended ten-node architecture."""
    validate_config(config)
    net = config['network']
    if config['system_type'] != 'fullsys' or len(config.get('limbs', [])) != 1:
        raise ValueError('Нужна fullsys с одной конечностью')
    if (net['neuron_count'], net['input_size'], net['output_size'], net['afferent_size']) != (10, 2, 2, 6):
        raise ValueError('Размеры сети должны быть 10/2/2/6')
    expected = {(s, t, int(np.sign(w))) for s, t, w in PAPER_CONNECTIONS}
    connections = [tuple(edge) for edge in net['connectivity']]
    if len(connections) != len(expected) or set(connections) != expected:
        raise ValueError('Топология не соответствует схеме Markin 2010')
    mask = np.zeros((10, 10))
    for source, target, sign in expected:
        mask[target, source] = sign
    if not np.array_equal(np.sign(net['weights']), mask):
        raise ValueError('Изменены знаки связей или появились связи вне маски')
    for key, shape in [('Q_aff', (10, 6)), ('Q_app', (10, 2))]:
        expected_mask = np.zeros(shape, dtype=bool)
        if key == 'Q_aff':
            for source, target, _ in PAPER_AFFERENTS:
                expected_mask[target, source] = True
        else:
            for target, source in ((0, 0), (1, 0), (4, 0), (5, 0), (6, 1)):
                expected_mask[target, source] = True
        matrix = np.asarray(net[key])
        if np.any(matrix < 0) or not np.array_equal(matrix > 0, expected_mask):
            raise ValueError(f'Изменена маска входов {key}')
    expected_p = np.zeros((2, 10))
    expected_p[0, 8], expected_p[1, 9] = 1., 1.
    if not np.array_equal(net['P'], expected_p):
        raise ValueError('Мышцы должны получать сигналы только от Mn-F/E')
    for key in ('weights', 'tau_syn', 'Q_app', 'Q_aff', 'P'):
        if not np.isfinite(net[key]).all():
            raise ValueError(f'Неконечное значение: {key}')
    if np.any(np.asarray(net['tau_syn']) <= 0):
        raise ValueError('Синаптические tau должны быть положительными')
    for key, values in net['neuron_params'].items():
        if not np.isfinite(values).all() or np.any(np.asarray(values) * (-1 if key == 'c' else 1) <= 0):
            raise ValueError(f'Недопустимые параметры нейронов: {key}')
    limb = config['limbs'][0]
    mechanics = limb['mechanics']
    if not np.isfinite(list(mechanics.values())).all():
        raise ValueError('Механика и начальные условия должны содержать конечные числа')
    if not (mechanics['mass'] > 0 and mechanics['length'] > 0
            and mechanics['tendon_a1'] > mechanics['tendon_a2'] > 0):
        raise ValueError('Нужны положительные масса/длина и a1 > a2 > 0')
    for muscle in ('flexor', 'extensor'):
        if any(not np.isfinite(v) or v <= 0 for v in limb[muscle].values()):
            raise ValueError(f'Параметры мышцы {muscle} должны быть положительными')
    if not np.isfinite([config['simulation']['dt'], config['simulation']['duration']]).all():
        raise ValueError('Шаг и длительность должны быть конечными числами')
    if config['simulation']['input_current'] != {'type': 'constant', 'amplitude': 1., 'neurons': [0, 1]}:
        raise ValueError('Этот опыт использует два постоянных входа, равных 1; драйвы меняются в Q_app')


def build_config(system, genome):
    """Apply a genome to a copy of an API configuration, with fixed topology."""
    validate_system(system)
    config = deepcopy(system)
    allowed = _allowed_paths()
    for gene in _gene_records(genome):
        path = tuple(gene['path'])
        if path not in allowed:
            raise ValueError(f'Ген {gene["name"]} изменяет фиксированный параметр: {path}')
        node = config
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = float(gene['value'])
    validate_system(config)
    return config


def simulate(config, *, duration_s=None, feedback=True, progress=False, drive=1.):
    """Run a fresh SpikingNN Simulation; return native histories, time in ms."""
    config = deepcopy(config)
    if duration_s is not None:
        if not np.isfinite(duration_s) or duration_s <= 0:
            raise ValueError('Длительность должна быть положительным конечным числом')
        config['simulation']['duration'] = float(duration_s) * 1000
    validate_system(config)
    if not np.isfinite(drive) or drive < 0:
        raise ValueError('Тонический драйв должен быть конечным неотрицательным числом')
    for row in config['network']['Q_app']:
        row[0] *= drive
    if not feedback:
        config['network']['Q_aff'] = np.zeros((10, 6)).tolist()
    # A fresh object is essential: run_simulation itself continues the current state.
    data = run_simulation(Simulation(config), progress=progress).get_data()
    if 'limbs' not in data:
        raise RuntimeError('Нужна версия SpikingNN API с записью механики fullsys из этого checkout')
    arrays = [data['time'], data['V'], data['U']]
    arrays.extend(next(iter(data['limbs'].values())).values())
    if not all(np.isfinite(a).all() for a in arrays):
        raise FloatingPointError('Динамика содержит NaN/Inf; такой геном следует отвергать при оптимизации')
    return data


def summarize(data):
    """Compact numerical evidence; these diagnostics are not a fitness function."""
    limb = next(iter(data['limbs'].values()))
    last_half = slice(len(data['time']) // 2, None)
    q = np.rad2deg(limb['angle_q'][last_half])
    return {
        'finite': True,
        'samples': len(data['time']),
        'spikes_per_neuron': dict(zip(NEURONS, (data['V'] >= 30).sum(axis=0).tolist())),
        'angle_range_last_half_deg': [float(q.min()), float(q.max())],
        'max_muscle_force_N': [float(limb[k].max()) for k in ('force_flex', 'force_ext')],
    }


def _plot_network(ax):
    """Draw the fixed recurrent architecture; input projections are documented in JSON."""
    pos = np.array([[-1, 3.4], [1, 3.4], [-.62, 2.55], [.62, 2.55],
                    [-1, 1.7], [1, 1.7], [-.22, .95], [.72, .85], [-1, -.1], [1, -.1]])
    for source, target, weight in PAPER_CONNECTIONS:
        curvature = .12 if abs(pos[source, 0] - pos[target, 0]) > .8 else 0
        ax.add_patch(FancyArrowPatch(pos[source], pos[target], shrinkA=17, shrinkB=17,
                     arrowstyle='-|>', mutation_scale=10, connectionstyle=f'arc3,rad={curvature}',
                     color='#52697b' if weight > 0 else '#bd3b52', linewidth=1.2))
    for i, (x, y) in enumerate(pos):
        color = '#dcecf5' if i in (0, 2, 4, 8) else '#fbe4d6'
        if i in (6, 7):
            color = '#ececec'
        ax.add_patch(Circle((x, y), .29, facecolor=color, edgecolor='#586675', zorder=3))
        ax.text(x, y, NEURONS[i], ha='center', va='center', fontsize=8, zorder=4)
    ax.text(-1, 3.93, 'Flexor', color=FLEXOR, ha='center', weight='bold')
    ax.text(1, 3.93, 'Extensor', color=EXTENSOR, ha='center', weight='bold')
    ax.text(0, -.65, 'Blue-grey: excitation   |   Red: inhibition\nConstant drive + limb afferents enter the network',
            ha='center', va='top', fontsize=8)
    ax.set(xlim=(-1.65, 1.65), ylim=(-1.15, 4.25), aspect='equal')
    ax.axis('off')


def plot_dynamics(data, output, *, feedback, fingerprint, summary, genome_label='Untrained starting genome'):
    """Save one figure with the network and five panels of actual API histories."""
    t = data['time'] / 1000
    limb = next(iter(data['limbs'].values()))
    stride = max(1, len(t) // 12000)
    ix = slice(None, None, stride)
    q, velocity = np.rad2deg(limb['angle_q']), np.rad2deg(limb['angular_velocity_w'])
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(3, 2, figsize=(14, 12), layout='constrained')
        ax = axes[0, 0]
        _plot_network(ax)
        ax.set_title('A  Markin 2010 connectivity adapted to SpikingNN', loc='left', fontsize=11)
        ax = axes[0, 1]
        colors = [FLEXOR if i in (0, 2, 4, 8) else EXTENSOR for i in range(10)]
        colors[6] = '#777777'
        start = max(0., t[-1] - 2.)
        ax.eventplot([t[(data['V'][:, i] >= 30) & (t >= start)] for i in range(10)],
                     colors=colors, linewidths=.7, linelengths=.7)
        ax.set(yticks=range(10), yticklabels=NEURONS, ylim=(9.7, -.7), xlabel='Time (s)')
        ax.set_xlim(start, max(t[-1], start + .001))
        ax.set_title('B  Spikes: final 2 s (or full run if shorter)', loc='left', fontsize=11)
        ax = axes[1, 0]
        ax.plot(t[ix], limb['force_flex'][ix], color=FLEXOR, label='Flexor')
        ax.plot(t[ix], limb['force_ext'][ix], color=EXTENSOR, label='Extensor')
        ax.set(xlabel='Time (s)', ylabel='Muscle force (N)')
        ax.set_title('C  Antagonistic muscle forces', loc='left', fontsize=11)
        ax.legend(frameon=False, ncols=2)
        ax = axes[1, 1]
        ax.plot(t[ix], q[ix], color='#385f4f')
        ax.axhline(90, color='grey', linestyle=':', linewidth=.8, label='Downward vertical')
        ax.set(xlabel='Time (s)', ylabel='Joint angle q (deg)')
        ax.set_title('D  One-degree-of-freedom limb', loc='left', fontsize=11)
        ax.legend(frameon=False)
        ax = axes[2, 0]
        ax.plot(q[ix], velocity[ix], color='#c2c7ca', linewidth=.8, label='Full simulation')
        late = slice(len(t) // 2, None, stride)
        ax.plot(q[late], velocity[late], color='#385f4f', linewidth=1, label='Second half')
        ax.scatter(q[0], velocity[0], color='black', s=18, zorder=4, label='Initial state')
        ax.set(xlabel='Joint angle q (deg)', ylabel='Angular velocity (deg/s)')
        ax.set_title('E  Limb phase portrait', loc='left', fontsize=11)
        ax.legend(frameon=False, fontsize=8)
        ax = axes[2, 1]
        for i, name in enumerate(AFFERENTS):
            ax.plot(t[ix], limb['afferents'][ix, i], label=name,
                    color=FLEXOR if i < 3 else EXTENSOR, linestyle=('-', ':', '--')[i % 3], linewidth=1)
        ax.set(xlabel='Time (s)', ylabel='Afferent signal (API units)')
        ax.set_title('F  Sensory feedback from muscles and limb', loc='left', fontsize=11)
        ax.legend(frameon=False, ncols=3, fontsize=8)
        for ax in axes.flat:
            if ax.axison:
                ax.grid(alpha=.15)
        mode = 'closed loop' if feedback else 'feedback disabled'
        fig.suptitle('Spiking CPG + one-DOF limb | ' + mode + '\n' +
                     genome_label+'; constant input; topology adapted from Markin et al. (2010)', fontsize=13)
        fig.savefig(output, dpi=180, metadata={'Description': json.dumps(summary), 'Configuration-SHA256': fingerprint})
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=HERE / 'system.json', help='JSON системы SpikingNN')
    parser.add_argument('--genome', type=Path, default=HERE / 'genome.json', help='JSON генома параметров')
    parser.add_argument('--duration', type=float, default=None, metavar='SECONDS', help='Длительность в секундах; по умолчанию из system.json (10 s)')
    parser.add_argument('--output', type=Path, default=None, help='PNG; по умолчанию dynamics.png рядом со скриптом')
    parser.add_argument('--no-feedback', action='store_true', help='Отключить Q_aff для проверки автономного генератора ритма')
    parser.add_argument('--progress', action='store_true', help='Показывать прогресс расчёта API')
    parser.add_argument('--drive', type=float, default=None, help='Постоянный множитель входа d1; по умолчанию из обученного генома или 1')
    args = parser.parse_args()
    system = json.loads(args.config.read_text(encoding='utf-8'))
    genome = json.loads(args.genome.read_text(encoding='utf-8'))
    config = build_config(system, genome)
    if args.duration is not None:
        if not np.isfinite(args.duration) or args.duration <= 0:
            parser.error('--duration должен быть положительным конечным числом')
        config['simulation']['duration'] = args.duration * 1000
    output = args.output or HERE / ('dynamics_no_feedback.png' if args.no_feedback else 'dynamics.png')
    output = output.resolve()
    if output.suffix.lower() != '.png':
        parser.error('--output должен иметь расширение .png')
    if not output.parent.is_dir():
        parser.error(f'Каталог для изображения не существует: {output.parent}')
    feedback = not args.no_feedback
    drive = args.drive if args.drive is not None else genome.get('training', {}).get('task', {}).get('tonic_drive', 1.)
    if not np.isfinite(drive) or drive < 0:
        parser.error('--drive должен быть конечным неотрицательным числом')
    print(f'SpikingNN API: {len(genome["genes"])} genes, {config["simulation"]["duration"] / 1000:g} s, '
          f'dt={config["simulation"]["dt"]:g} ms, feedback={feedback}, drive={drive:g}', flush=True)
    data = simulate(config, feedback=feedback, progress=args.progress, drive=drive)
    summary = summarize(data)
    summary.update(duration_s=config['simulation']['duration'] / 1000, dt_ms=config['simulation']['dt'], feedback=feedback, tonic_drive=drive)
    effective = deepcopy(config)
    for row in effective['network']['Q_app']:
        row[0] *= drive
    if not feedback:
        effective['network']['Q_aff'] = np.zeros((10, 6)).tolist()
    fingerprint = hashlib.sha256(json.dumps(effective, sort_keys=True).encode()).hexdigest()
    label='GA-trained genome' if 'training' in genome else 'Untrained starting genome'
    plot_dynamics(data, output, feedback=feedback, fingerprint=fingerprint, summary=summary, genome_label=label)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Figure: {output}')


if __name__ == '__main__':
    main()
