"""Запуск четырёхконечностной системы на постоянных токах и построение фигуры.

Верхняя панель: растр всех 16 нейронов. Четыре средние панели: силы
сгибателя и разгибателя каждой конечности. Нижняя панель: четыре угла
суставов на общей оси. Цвет конечности одинаков во всех панелях.
CPG-параметры загружаются из network.neuron_params: a=0.003 у восьми
CPG-нейронов; параметры восьми мотонейронов соответствуют пресету FS.

Из корня проекта:
    .venv/bin/python examples/4Limb_cpg_system/costume_neurons/build_config.py
    .venv/bin/python examples/4Limb_cpg_system/costume_neurons/run_simulation.py --progress

Сохранить рисунок без открытия окна:
    .venv/bin/python examples/4Limb_cpg_system/costume_neurons/run_simulation.py --no-show

Другой постоянный ток и длительность (nA и мс):
    .venv/bin/python examples/4Limb_cpg_system/costume_neurons/run_simulation.py --current 10 --duration 5000

Нарисовать только выбранный диапазон секунд после полной симуляции:
    .venv/bin/python examples/4Limb_cpg_system/costume_neurons/run_simulation.py --time 1:2
    .venv/bin/python examples/4Limb_cpg_system/costume_neurons/run_simulation.py --duration 180000 --time 150:180 --progress

JSON загружается публичным API. Растр рассчитывает существующая функция
SpikingNN.utils.net_preparation.create_firing_rastr. Ток не содержит шума
или периодической модуляции. Состояния записываются до каждого шага.
"""

import argparse
from pathlib import Path

import numpy as np

from SpikingNN import create_simulation, get_results, run_simulation
from SpikingNN.schema import validate_config
from SpikingNN.utils.net_preparation import create_firing_rastr

from build_config import DEFAULT_CONFIG_PATH, NEURON_ROLES


def parse_time_range(value):
    """Прочитать диапазон START:END в секундах для аргумента --time."""
    try:
        start, end = (float(part) for part in value.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError("Ожидается диапазон START:END в секундах, например 150:180") from None
    if not np.isfinite([start, end]).all() or not 0 <= start < end:
        raise argparse.ArgumentTypeError("Диапазон должен удовлетворять 0 <= START < END; границы должны быть конечными")
    return start, end


def plot_dynamics(data, neuron_names, spike_threshold=30, time_range=None):
    """Создать фигуру для всего времени или диапазона time_range в секундах."""
    import matplotlib.pyplot as plt

    time_seconds = data["time"] / 1000.0
    window = slice(None)
    if time_range is not None:
        start, end = time_range
        window = (time_seconds >= start) & (time_seconds <= end)
        if not window.any():
            raise ValueError("Выбранный диапазон времени не содержит записанных шагов")
    time_seconds = time_seconds[window]
    spike_times, spike_neurons = create_firing_rastr(data["V"][window], data["time"][window], spike_threshold)
    names = list(data["limbs"])
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    figure = plt.figure(figsize=(14, 12), layout="constrained")
    grid = figure.add_gridspec(4, 2, height_ratios=(1.6, 1, 1, 1.2))
    raster = figure.add_subplot(grid[0, :])
    for i, (name, color) in enumerate(zip(names, colors)):
        selected = (spike_neurons // 4) == i
        raster.scatter(spike_times[selected] / 1000.0, spike_neurons[selected],
                       marker="|", s=18, linewidths=0.6, color=color, label=name)
        if i:
            raster.axhline(4 * i - 0.5, color="0.8", linewidth=0.7)
    raster.set_yticks(range(16), neuron_names, fontsize=8)
    raster.set_ylim(15.5, -0.5)
    raster.set_title("Растр спайковой активности")
    raster.set_ylabel("Нейрон")
    raster.legend(loc="upper right", ncol=4)

    angles = figure.add_subplot(grid[3, :], sharex=raster)
    styles = ("-", "--", "-.", ":")
    for i, (name, color) in enumerate(zip(names, colors)):
        history = data["limbs"][name]
        muscles = figure.add_subplot(grid[1 + i // 2, i % 2], sharex=raster)
        muscles.plot(time_seconds, history["force_flex"][window], color=color, label="Сгибатель")
        muscles.plot(time_seconds, history["force_ext"][window], color=color, linestyle="--", label="Разгибатель")
        muscles.set_title(f"Мышцы {name}")
        muscles.set_ylabel("Сила, Н")
        muscles.legend(loc="upper right")
        muscles.grid(alpha=0.25)
        muscles.set_xlabel("Время, с")
        angles.plot(time_seconds, history["angle_q"][window], color=color, linestyle=styles[i], label=name)
    angles.set_title("Углы четырёх конечностей")
    angles.set_ylabel("Угол q, рад")
    angles.set_xlabel("Время, с")
    angles.legend(ncol=4)
    angles.grid(alpha=0.25)
    raster.set_xlim(*(time_range if time_range is not None else (time_seconds[0], time_seconds[-1])))
    raster.set_xlabel("Время, с")
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--current", type=float, help="Постоянный ток каждого входного канала, nA")
    parser.add_argument("--duration", type=float, help="Длительность, мс")
    parser.add_argument("--time", type=parse_time_range, metavar="START:END",
                        help="Диапазон отрисовки в секундах, например 150:180; симуляция выполняется целиком")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().with_name("dynamics.png"))
    parser.add_argument("--no-show", action="store_true", help="Сохранить рисунок без открытия окна")
    parser.add_argument("--progress", action="store_true", help="Показать прогресс симуляции (tqdm)")
    args = parser.parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    simulation = create_simulation(args.config)
    if simulation.network.N != 16 or len(simulation.limbs) != 4:
        parser.error("Пример ожидает 16 нейронов и 4 конечности")
    if args.duration is not None:
        simulation.config["simulation"]["duration"] = args.duration
    validate_config(simulation.config)
    if args.time is not None:
        start, end = args.time
        duration = simulation.config["simulation"]["duration"]
        dt = simulation.config["simulation"]["dt"]
        if end > duration / 1000.0:
            parser.error(f"Диапазон --time выходит за длительность {duration / 1000:g} с; "
                         "увеличьте --duration (в мс)")
        first_sample = np.ceil(start * 1000.0 / dt) * dt
        if first_sample >= duration or first_sample > end * 1000.0:
            parser.error("Диапазон --time не содержит шагов при выбранном dt")
    signal = dict(simulation.config["simulation"].get("input_current", {}))
    if signal.get("type") != "constant":
        parser.error("В конфигурации должен быть постоянный входной ток")
    if args.current is not None:
        signal["amplitude"] = args.current
    names = [f"{limb}.{role}" for limb in simulation.system.limb_names for role in NEURON_ROLES]
    for i, name in enumerate(names):
        simulation.network.set_name(i, name)

    print(f"Запуск: I = {signal['amplitude']:g} nA, "
          f"T = {simulation.config['simulation']['duration']:g} мс, "
          f"dt = {simulation.config['simulation']['dt']:g} мс")
    data = get_results(run_simulation(simulation, signal, progress=args.progress))
    arrays = [data["V"], data["U"]]
    arrays.extend(values for limb in data["limbs"].values() for values in limb.values())
    if not all(np.isfinite(values).all() for values in arrays):
        raise RuntimeError("В динамике появились NaN/Inf; проверьте токи, веса и шаг времени")
    spike_times, spike_neurons = create_firing_rastr(data["V"], data["time"], simulation.network.V_peak)
    counts = np.bincount(spike_neurons, minlength=16)
    print(f"Записано {len(data['time'])} шагов; всего {len(spike_times)} спайков")
    for i, name in enumerate(simulation.system.limb_names):
        angle = data["limbs"][name]["angle_q"]
        print(f"{name}: спайки [CPG_F, MN_F, CPG_E, MN_E] = {counts[4*i:4*i+4].tolist()}, "
              f"q = [{angle.min():.3f}, {angle.max():.3f}] рад")

    figure = plot_dynamics(data, names, simulation.network.V_peak, time_range=args.time)
    figure.suptitle(f"4 связанных CPG: постоянный ток входов {signal['amplitude']:g} nA")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(f"График: {args.output.resolve()}")
    import matplotlib.pyplot as plt
    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
