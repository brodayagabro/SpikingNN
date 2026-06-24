#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command Line Interface for SpikingNN
Cross-platform entry point for GUI application and simulation runner
"""

import sys
import subprocess
import os
import json
import argparse
from pathlib import Path

def get_app_path():
    """Получить путь к app.py внутри пакета"""
    return str(Path(__file__).parent / "GUI.py")

def run_simulation_cli(args):
    """Запуск симуляции из командной строки"""
    from SpikingNN.api import create_simulation, run_simulation, get_results
    
    # Загрузка конфигурации
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"❌ Ошибка: Файл конфигурации не найден: {args.config}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Ошибка: Невалидный JSON в файле конфигурации: {e}")
        sys.exit(1)
    
    # Загрузка параметров сигналов из аргументов командной строки
    signals = {}
    if args.signal_type:
        signals["type"] = args.signal_type
    if args.amplitude is not None:
        signals["amplitude"] = args.amplitude
    if args.frequency is not None:
        signals["frequency"] = args.frequency
    if args.neurons:
        signals["neurons"] = [int(n) for n in args.neurons.split(",")]
    
    # Создание симуляции
    try:
        sim = create_simulation(args.config)
        print(f"✅ Симуляция создана: {sim.system_type} с {sim.network.N} нейронами")
    except Exception as e:
        print(f"❌ Ошибка создания симуляции: {e}")
        sys.exit(1)
    
    # Запуск симуляции
    try:
        print("🚀 Запуск симуляции...")
        results = run_simulation(sim, signals)
        data = get_results(results)
        print(f"✅ Симуляция завершена!")
        print(f"   Временные шаги: {len(data['time'])}")
        print(f"   V shape: {data['V'].shape}")
        print(f"   U shape: {data['U'].shape}")
    except Exception as e:
        print(f"❌ Ошибка симуляции: {e}")
        sys.exit(1)
    
    # Сохранение результатов
    if args.output:
        output_files = args.output if isinstance(args.output, list) else [args.output]
        
        for output_file in output_files:
            try:
                # Определяем расширение файла
                if output_file.endswith('.csv'):
                    # Сохранение в CSV
                    import pandas as pd
                    df_v = pd.DataFrame(data["V"], columns=[f"V_{i}" for i in range(data["V"].shape[1])])
                    df_u = pd.DataFrame(data["U"], columns=[f"U_{i}" for i in range(data["U"].shape[1])])
                    df_time = pd.DataFrame({"time": data["time"]})
                    
                    # Объединяем все данные
                    df = pd.concat([df_time, df_v, df_u], axis=1)
                    df.to_csv(output_file, index=False)
                    print(f"💾 Результаты сохранены в {output_file}")
                elif output_file.endswith('.npz'):
                    # Сохранение в NumPy format
                    np.savez(output_file, 
                            time=data["time"], 
                            V=data["V"], 
                            U=data["U"])
                    print(f"💾 Результаты сохранены в {output_file}")
                else:
                    print(f"⚠️  Неизвестный формат файла: {output_file}")
            except Exception as e:
                print(f"❌ Ошибка сохранения в {output_file}: {e}")
    
    return data

def main():
    """Точка входа для команды spikingnn"""
    parser = argparse.ArgumentParser(
        prog="spikingnn",
        description="🧠 Spiking Neural Network Simulator - Модель Izhikevich",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  spikingnn gui                           Запустить GUI приложение (порт 8501)
  spikingnn gui --port 8502               Запустить GUI на порту 8502
  spikingnn gui --host 0.0.0.0            Запустить GUI для доступа из сети
  spikingnn gui --debug                   Запустить GUI в режиме отладки
  
  spikingnn sim config.json               Запустить симуляцию с конфигурацией
  spikingnn sim config.json -o result.csv Сохранить результаты в CSV
  spikingnn sim config.json --signal-type sine --amplitude 10 --frequency 1
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Команды')
    
    # Подкоманда GUI
    gui_parser = subparsers.add_parser('gui', help='Запустить GUI приложение')
    gui_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8501,
        help="Порт для веб-сервера (по умолчанию: 8501)"
    )
    gui_parser.add_argument(
        "--host", "-H",
        type=str,
        default="localhost",
        help="Хост для веб-сервера (по умолчанию: localhost)"
    )
    gui_parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Запустить в режиме отладки"
    )
    gui_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Не открывать браузер автоматически"
    )
    
    # Подкоманда SIM
    sim_parser = subparsers.add_parser('sim', help='Запустить симуляцию')
    sim_parser.add_argument(
        "config",
        type=str,
        help="Путь к файлу конфигурации JSON"
    )
    sim_parser.add_argument(
        "--output", "-o",
        type=str,
        nargs='+',
        help="Пути для сохранения результатов (CSV или NPZ)"
    )
    sim_parser.add_argument(
        "--signal-type",
        type=str,
        choices=['constant', 'sine', 'square', 'ramp', 'noise'],
        help="Тип сигнала"
    )
    sim_parser.add_argument(
        "--amplitude",
        type=float,
        help="Амплитуда сигнала (nA)"
    )
    sim_parser.add_argument(
        "--frequency",
        type=float,
        help="Частота сигнала (Hz) для периодических сигналов"
    )
    sim_parser.add_argument(
        "--neurons",
        type=str,
        help="Индексы нейронов через запятую (например: 0,1,2)"
    )
    
    # Общие аргументы
    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Показать версию пакета"
    )
    
    args = parser.parse_args()
    
    # Обработка --version
    if args.version:
        try:
            from SpikingNN import __version__
            print(f"SpikingNN v{__version__}")
        except ImportError:
            print("SpikingNN v0.0.2")
        sys.exit(0)
    
    # Если команда не указана, показываем справку
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # Выполнение команды
    if args.command == 'gui':
        # Получить путь к приложению
        app_path = get_app_path()
        
        if not os.path.exists(app_path):
            print(f"❌ Ошибка: app.py не найден по пути {app_path}")
            sys.exit(1)
        
        # Сформировать команду streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            app_path,
            "--server.port", str(args.port),
            "--server.address", args.host,
            "--server.headless", "true" if args.no_browser else "false"
        ]
        
        if args.debug:
            cmd.extend(["--logger.level", "debug"])
        
        print(f"🚀 Запуск SpikingNN GUI на http://{args.host}:{args.port}")
        print(f"📍 Приложение: {app_path}")
        print("ℹ️  Нажмите Ctrl+C для остановки\n")
        
        # Запустить Streamlit
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка запуска: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n👋 Приложение остановлено пользователем")
            sys.exit(0)
    
    elif args.command == 'sim':
        run_simulation_cli(args)

if __name__ == "__main__":
    main()