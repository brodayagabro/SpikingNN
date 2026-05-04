# tests/test_cross_module_imports.py
import sys
import subprocess
import pytest
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

def test_core_and_models_import_chain():
    """
    Verify that models can import from core without circular dependency errors.
    Проверяет, что модуль models корректно импортирует core без циклических зависимостей.
    """
    code = """
import sys
try:
    from SpikingNN.models.Var_Limb import Var_Limb
    from SpikingNN.core.Izh_net import Izhikevich_Network
    print("OK")
except ImportError as e:
    print(f"FAIL: {e}")
    sys.exit(1)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"Import chain broken:\n{result.stderr}"
    assert "OK" in result.stdout

def test_visualize_neuromechanical_loop(tmp_path):
    """
    Generate a block diagram of the closed-loop neuromechanical system.
    Генерирует блок-схему замкнутой нейромеханической системы для CI-валидации.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    ax.set_title("Neuromechanical Closed Loop", fontsize=12, pad=10)
    
    blocks = {"CPG Network": (0.25, 0.5), "Muscles": (0.5, 0.75), "Limb": (0.5, 0.25), "Afferents": (0.75, 0.5)}
    for name, (x, y) in blocks.items():
        ax.annotate(name, (x, y), ha="center", va="center", fontsize=10,
                    bbox=dict(boxstyle="round", fc="#2196F3", ec="white"))
    
    ax.annotate("", xy=(0.5, 0.75), xytext=(0.25, 0.5), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.5, 0.25), xytext=(0.5, 0.75), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.75, 0.5), xytext=(0.5, 0.25), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.25, 0.5), xytext=(0.75, 0.5), arrowprops=dict(arrowstyle="->", lw=2, ls="--", color="red"))
    
    out = tmp_path / "loop_diagram.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    assert out.exists()