#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command Line Interface for SpikingNN
Cross-platform entry point for GUI application
"""

import sys
import subprocess
import os
from pathlib import Path

def get_app_path():
    """Получить путь к app.py внутри пакета"""
    return str(Path(__file__).parent / "GUI.py")

def main():
    """Точка входа для команды spikingnn"""
    import argparse
    
    parser = argparse.ArgumentParser(
        prog="spikingnn",
        description="🧠 Spiking Neural Network Simulator - Модель Izhikevich",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  spikingnn                    Запустить приложение (порт 8501)
  spikingnn --port 8502        Запустить на порту 8502
  spikingnn --host 0.0.0.0     Запустить для доступа из сети
  spikingnn --debug            Запустить в режиме отладки
  spikingnn --version          Показать версию
        """
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8501,
        help="Порт для веб-сервера (по умолчанию: 8501)"
    )
    
    parser.add_argument(
        "--host", "-H",
        type=str,
        default="localhost",
        help="Хост для веб-сервера (по умолчанию: localhost)"
    )
    
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Запустить в режиме отладки"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Показать версию пакета"
    )
    
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Не открывать браузер автоматически"
    )
    
    args = parser.parse_args()
    
    # Обработка --version
    if args.version:
        try:
            from SpikingNN import __version__
            print(f"SpikingNN v{__version__}")
        except ImportError:
            print("SpikingNN v1.0.0")
        sys.exit(0)
    
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
    
    print(f"🚀 Запуск SpikingNN на http://{args.host}:{args.port}")
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

if __name__ == "__main__":
    main()