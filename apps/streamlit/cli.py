#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command Line Interface for SpikingNN Streamlit GUI
Cross-platform entry point for launching the web application.

Запуск веб-интерфейса SpikingNN на базе Streamlit.
Кроссплатформенная точка входа для запуска веб-приложения.
"""

import sys
import subprocess
import os
from pathlib import Path


def get_app_path() -> str:
    """
    Get the absolute path to the Streamlit application file.

    Returns
    -------
    str
        Absolute path to GUI.py within the apps/streamlit/ directory.

    Получить абсолютный путь к файлу Streamlit-приложения.

    Возвращает
    ----------
    str
        Абсолютный путь к GUI.py внутри директории apps/streamlit/.
    """
    return str(Path(__file__).parent / "GUI.py")


def main() -> None:
    """
    Entry point for the 'spknn-gui' command.
    Parses arguments and launches Streamlit with specified options.

    Точка входа для команды 'spknn-gui'.
    Парсит аргументы и запускает Streamlit с указанными опциями.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        prog="spknn-gui",
        description="Spiking Neural Network Simulator - Streamlit Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  spknn-gui                    Launch app on port 8501
  spknn-gui --port 8502        Launch on custom port
  spknn-gui --host 0.0.0.0     Allow network access
  spknn-gui --debug            Enable debug logging
  spknn-gui --version          Show package version

Примеры:
  spknn-gui                    Запустить приложение (порт 8501)
  spknn-gui --port 8502        Запустить на порту 8502
  spknn-gui --host 0.0.0.0     Разрешить доступ из сети
  spknn-gui --debug            Режим отладки
  spknn-gui --version          Показать версию
        """
    )
    
    parser.add_argument("--port", "-p", type=int, default=8501,
                        help="Web server port (default: 8501) / Порт веб-сервера")
    parser.add_argument("--host", "-H", type=str, default="localhost",
                        help="Bind address (default: localhost) / Адрес привязки")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Enable debug logging / Включить отладочный лог")
    parser.add_argument("--version", "-v", action="store_true",
                        help="Show package version / Показать версию пакета")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open browser automatically / Не открывать браузер")
    
    args = parser.parse_args()
    
    # Handle --version
    if args.version:
        try:
            from SpikingNN import __version__
            print(f"SpikingNN v{__version__}")
        except ImportError:
            print("SpikingNN v0.2.0")
        sys.exit(0)
    
    # Get application path
    app_path = get_app_path()
    
    if not os.path.exists(app_path):
        print(f"❌ Error: GUI.py not found at {app_path}")
        sys.exit(1)
    
    # Build Streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        app_path,
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--server.headless", "true" if args.no_browser else "false"
    ]
    
    if args.debug:
        cmd.extend(["--logger.level", "debug"])
    
    print(f"🚀 Launching SpikingNN at http://{args.host}:{args.port}")
    print(f"📍 Application: {app_path}")
    print("ℹ️  Press Ctrl+C to stop\n")
    
    # Launch Streamlit subprocess
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Launch error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        sys.exit(0)


if __name__ == "__main__":
    main()