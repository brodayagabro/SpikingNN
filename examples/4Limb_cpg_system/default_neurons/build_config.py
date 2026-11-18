"""Сборка четырёх связанных CPG-модулей и экспорт конфигурации в JSON.

В этом примере каждый полуцентровый модуль (HCO) содержит четыре нейрона:
CPG_F (CH), MN_F (FS), CPG_E (CH), MN_E (FS). CPG_F и CPG_E взаимно
тормозят друг друга и возбуждают соответствующие мотонейроны. MN_F и MN_E
управляют сгибателем и разгибателем одной конечности.

Порядок модулей: FL — передняя левая, FR — передняя правая,
HL — задняя левая, HR — задняя правая. Индексы нейронов модуля i:
4*i + [0, 1, 2, 3] = [CPG_F, MN_F, CPG_E, MN_E].

Диагональные пары FL–HR и FR–HL связаны возбуждением одноимённых CPG.
Контралатеральные и ипсилатеральные пары связаны торможением одноимённых
CPG. Такая топология задаёт предпочтение диагональной координации;
конкретный режим движения зависит от токов и весов, а не гарантируется
одной топологией.

Матрицы используют соглашение weights[target, source]. Знак связи явно
записан в weights, а connectivity повторяет её знак. Все отсутствующие
связи имеют нулевой вес. Размеры: weights и tau_syn — 16x16,
Q_app — 16x4, Q_aff — 16x24, P — 8x16.
В JSON каждая строка матрицы записывается на отдельной строке целиком.

Каждый из четырёх входных каналов получает постоянный ток current (nA).
Q_app направляет его на оба CPG соответствующей конечности. Коэффициенты
1.0 и 0.98 слегка нарушают симметрию F/E; у диагональных пар предпочтение
одинаково. Мотонейроны прямой внешней стимуляции не получают.

Афференты каждой конечности идут только в её модуль в порядке
[Ia_F, II_F, Ib_F, Ia_E, II_E, Ib_E]. Они тормозят соответствующий CPG
и возбуждают мотонейрон через Ia/II; Ib тормозит мотонейрон.
--afferent-gain 0 отключает эту обратную связь.

Единицы: время и tau — мс, угол — рад, длина — м, масса — кг,
сила — Н. Модели и численная логика core используются без изменений.

Запуск из корня проекта:
    .venv/bin/python examples/4Limb_cpg_system/default_neurons/build_config.py
"""

import argparse
import json
from pathlib import Path

import numpy as np

from SpikingNN.schema import validate_config


LIMB_NAMES = ("FL", "FR", "HL", "HR")
NEURON_ROLES = ("CPG_F", "MN_F", "CPG_E", "MN_E")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("config.json")


def build_config(duration=3000.0, dt=0.1, current=8.0, coupling=0.1, afferent_gain=0.1):
    """Вернуть JSON-совместимый словарь для публичного API SpikingNN."""
    for name, value in (("duration", duration), ("dt", dt), ("current", current),
                        ("coupling", coupling), ("afferent_gain", afferent_gain)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if duration == 0 or dt < 0.001:
        raise ValueError("duration must be positive and dt must be at least 0.001 ms")

    n = 4 * len(LIMB_NAMES)
    weights = np.zeros((n, n))
    tau_syn = np.full((n, n), 20.0)
    q_app = np.zeros((n, 4))
    q_aff = np.zeros((n, 24))
    projection = np.zeros((8, n))
    connectivity = []

    def connect(source, target, weight, tau):
        if weight == 0:
            return
        weights[target, source] = weight
        tau_syn[target, source] = tau
        connectivity.append([source, target, int(np.sign(weight))])

    for i in range(4):
        f, mn_f, e, mn_e = (4 * i + j for j in range(4))
        connect(f, mn_f, 1.5, 10.0)
        connect(e, mn_e, 1.5, 10.0)
        connect(f, e, -1.5, 20.0)
        connect(e, f, -1.5, 20.0)

        prefer_flexor = i in (0, 3)
        q_app[f, i] = 1.0 if prefer_flexor else 0.98
        q_app[e, i] = 0.98 if prefer_flexor else 1.0
        projection[2 * i, mn_f] = 1.0
        projection[2 * i + 1, mn_e] = 1.0

        # Local feedback: [Ia_F, II_F, Ib_F, Ia_E, II_E, Ib_E].
        q_aff[4 * i:4 * i + 4, 6 * i:6 * i + 6] = afferent_gain * np.array([
            [-1, -1, 0, 0, 0, 0],
            [1, 1, -1, 0, 0, 0],
            [0, 0, 0, -1, -1, 0],
            [0, 0, 0, 1, 1, -1],
        ])

    diagonal_pairs = ((0, 3), (1, 2))
    neighboring_pairs = ((0, 1), (2, 3), (0, 2), (1, 3))
    for pairs, sign in ((diagonal_pairs, 1), (neighboring_pairs, -1)):
        for left, right in pairs:
            for role in (0, 2):
                connect(4 * left + role, 4 * right + role, sign * coupling, 20.0)
                connect(4 * right + role, 4 * left + role, sign * coupling, 20.0)

    config = {
        "system_type": "fullsys",
        "network": {
            "neuron_count": n,
            "neuron_types": ["CH", "FS", "CH", "FS"] * 4,
            "input_size": 4,
            "output_size": 8,
            "afferent_size": 24,
            "connectivity": connectivity,
            "weights": weights.tolist(),
            "tau_syn": tau_syn.tolist(),
            "Q_app": q_app.tolist(),
            "Q_aff": q_aff.tolist(),
            "P": projection.tolist(),
        },
        "simulation": {
            "dt": dt,
            "duration": duration,
            "input_current": {"type": "constant", "amplitude": current, "neurons": [0, 1, 2, 3]},
        },
        "limbs": [
            {
                "name": name,
                "mechanics": {
                    "mass": 0.3, "length": 0.3, "viscosity": 0.01,
                    "q0": float(np.pi / 2), "w0": 0.0,
                    "tendon_a1": 0.06, "tendon_a2": 0.007,
                },
                "flexor": {"w": 1.5, "N": 30, "A": 0.0074, "tau_c": 39, "tau_1": 21},
                "extensor": {"w": 1.5, "N": 30, "A": 0.0074, "tau_c": 39, "tau_1": 21},
            }
            for name in LIMB_NAMES
        ],
    }
    validate_config(config)
    return config


def format_config(config):
    """Форматировать JSON с отступами и компактными строками матриц."""
    def format_value(value, level):
        indent = "  " * level
        child_indent = "  " * (level + 1)
        if isinstance(value, dict) and value:
            entries = [
                f"{child_indent}{json.dumps(key, ensure_ascii=False)}: {format_value(item, level + 1)}"
                for key, item in value.items()
            ]
            return "{\n" + ",\n".join(entries) + "\n" + indent + "}"
        if isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value):
            entries = [child_indent + format_value(item, level + 1) for item in value]
            return "[\n" + ",\n".join(entries) + "\n" + indent + "]"
        return json.dumps(value, ensure_ascii=False, allow_nan=False)

    return format_value(config, 0) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--duration", type=float, default=3000.0, help="Длительность, мс")
    parser.add_argument("--dt", type=float, default=0.1, help="Шаг, мс")
    parser.add_argument("--current", type=float, default=8.0, help="Постоянный ток каждого входного канала, nA")
    parser.add_argument("--coupling", type=float, default=0.1, help="Модуль веса межмодульных связей")
    parser.add_argument("--afferent-gain", type=float, default=0.1, help="Масштаб локальной обратной связи")
    args = parser.parse_args()
    config = build_config(args.duration, args.dt, args.current, args.coupling, args.afferent_gain)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_config(config), encoding="utf-8")
    print(f"Конфигурация: {args.output.resolve()}")
    print(f"16 нейронов, 4 конечности, {len(config['network']['connectivity'])} направленных связей")


if __name__ == "__main__":
    main()
