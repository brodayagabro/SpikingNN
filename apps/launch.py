#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified GUI Launcher for SpikingNN
Launch either Streamlit or Tkinter interface via command-line flag.

Единый лаунчер графических интерфейсов SpikingNN.
Запускает Streamlit или Tkinter интерфейс через флаг командной строки.
"""

import argparse
import sys
import subprocess
from pathlib import Path


def _run_streamlit(port: int, host: str, debug: bool, no_browser: bool) -> int:
    """
    Launch Streamlit GUI in a subprocess with specified options.

    Parameters
    ----------
    port : int
        Network port for Streamlit server.
    host : str
        Bind address for the server.
    debug : bool
        Enable debug logging level.
    no_browser : bool
        Prevent automatic browser opening.

    Returns
    -------
    int
        Exit code from the Streamlit process.

    Запустить Streamlit-интерфейс в подпроцессе с указанными опциями.

    Параметры
    ----------
    port : int
        Сетевой порт для сервера Streamlit.
    host : str
        Адрес привязки сервера.
    debug : bool
        Включить отладочный уровень логирования.
    no_browser : bool
        Запретить автоматическое открытие браузера.

    Возвращает
    ----------
    int
        Код выхода процесса Streamlit.
    """
    gui_script = Path(__file__).parent / "streamlit" / "GUI.py"
    if not gui_script.exists():
        raise FileNotFoundError(f"Streamlit GUI not found: {gui_script}")
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(gui_script),
        "--server.port", str(port),
        "--server.address", host,
        "--server.headless", "true" if no_browser else "false"
    ]
    if debug:
        cmd.extend(["--logger.level", "debug"])
    
    return subprocess.run(cmd).returncode


def _run_tkinter() -> int:
    """
    Launch Tkinter GUI in a subprocess.

    Returns
    -------
    int
        Exit code from the Tkinter process.

    Запустить Tkinter-интерфейс в подпроцессе.

    Возвращает
    ----------
    int
        Код выхода процесса Tkinter.
    """
    gui_script = Path(__file__).parent / "tkinter" / "tkinter_app.py"
    if not gui_script.exists():
        raise FileNotFoundError(f"Tkinter GUI not found: {gui_script}")
    
    return subprocess.run([sys.executable, str(gui_script)]).returncode


def main() -> None:
    """
    Parse CLI arguments and dispatch to selected GUI framework.

    Supports two interface backends:
    - streamlit: Web-based interactive dashboard with Plotly visualizations
    - tkinter: Desktop application with native widgets

    Streamlit-specific flags (--port, --host, --debug) are ignored
    when Tkinter mode is selected.

    Парсит аргументы командной строки и запускает выбранный графический интерфейс.

    Поддерживает два бэкенда интерфейса:
    - streamlit: Веб-панель с интерактивными графиками Plotly
    - tkinter: Десктопное приложение с нативными виджетами

    Флаги, специфичные для Streamlit (--port, --host, --debug),
    игнорируются при выборе режима Tkinter.

    Raises
    ------
    FileNotFoundError
        If the selected GUI script is missing from apps/ directory.
    KeyboardInterrupt
        If user interrupts the process with Ctrl+C.

    Исключения
    ----------
    FileNotFoundError
        Если выбранный скрипт интерфейса отсутствует в директории apps/.
    KeyboardInterrupt
        Если пользователь прерывает процесс сочетанием Ctrl+C.
    """
    parser = argparse.ArgumentParser(
        prog="spknn-gui",
        description="Launch SpikingNN graphical interface"
    )
    parser.add_argument(
        "--gui", choices=["streamlit", "tkinter"], default="streamlit",
        help="GUI framework to launch (default: streamlit)"
    )
    # Streamlit-specific options
    parser.add_argument("--port", "-p", type=int, default=8501)
    parser.add_argument("--host", "-H", type=str, default="localhost")
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--version", "-v", action="store_true")
    
    args = parser.parse_args()
    
    if args.version:
        try:
            from SpikingNN import __version__
            print(f"SpikingNN v{__version__}")
        except ImportError:
            print("SpikingNN v0.2.0")
        sys.exit(0)
    
    print(f"🚀 Launching {args.gui.upper()} GUI...")
    
    try:
        if args.gui == "streamlit":
            exit_code = _run_streamlit(
                port=args.port, host=args.host,
                debug=args.debug, no_browser=args.no_browser
            )
        else:  # tkinter
            exit_code = _run_tkinter()
        sys.exit(exit_code)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
        sys.exit(0)


if __name__ == "__main__":
    main()