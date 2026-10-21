"""Resolve relocated results and load their unmodified numerical source snapshots.

Live commands use the AfferentBalance2026 package. Older saved runs retain their
SteppingReflexGA namespace and original source hashes. A process must choose its
numerical runtime before importing the neuron library or simulation modules.
"""
import hashlib
import importlib
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
LEGACY_PACKAGE = "SteppingReflexGA"
CURRENT_PACKAGE = "AfferentBalance2026"
_active_runtime = None
_core_package = f"scripts.{CURRENT_PACKAGE}.experiments.core"


def resolve_recorded_path(path, *, relative_to=None):
    """Resolve a saved path after relocation without rewriting run metadata.

    Relative paths are anchored at ``relative_to`` (the repository by default).
    Only the outer experiment directory is remapped; nested snapshot package
    names remain unchanged. The returned path need not exist.
    """
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path(relative_to or REPO) / path
    if path.exists():
        return path.resolve()
    parts = path.parts
    for index, part in enumerate(parts[:-1]):
        if part == "scripts" and parts[index + 1] in (LEGACY_PACKAGE, CURRENT_PACKAGE):
            relative = parts[index + 2:]
            candidate = BASE.joinpath(*relative)
            # Old experiment outputs moved intact; nested frozen namespaces stay intact.
            if (len(relative) >= 2 and relative[0] == "experiments"
                    and relative[1] in ("equilibrium", "small_nm_weights", "angle_setpoint")
                    and not candidate.exists()):
                candidate = BASE / "archive/pre_ABC_20261020" / Path(*relative)
            return candidate.resolve()
    return path.resolve()


def current_results(pointer=None):
    """Read the current-series pointer, accepting relative or historical paths."""
    pointer = Path(pointer) if pointer else BASE / "report/results/current_unscaled.json"
    record = json.loads(pointer.read_text())
    root = resolve_recorded_path(record["root"], relative_to=pointer.parent)
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(f"Result manifest missing: {root / 'manifest.json'}")
    return root


def runtime_directory(root):
    """Locate a series' saved runtime using its unchanged provenance manifest."""
    root = resolve_recorded_path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    runtime = resolve_recorded_path(manifest["runtime"], relative_to=root)
    if not runtime.is_dir():
        raise FileNotFoundError(f"Frozen numerical runtime missing: {runtime}")
    return runtime


def runtime_package_directory(runtime):
    """Find the current or historical experiment package inside a snapshot."""
    runtime = Path(runtime).resolve()
    candidates = [runtime / "scripts" / name for name in (CURRENT_PACKAGE, LEGACY_PACKAGE)]
    existing = [path for path in candidates if (path / "experiments/core").is_dir()]
    if len(existing) != 1:
        raise ValueError(f"Expected exactly one numerical experiment package: {runtime}")
    return existing[0]


def recorded_runtime(result):
    """Find the exact runtime referenced by a saved run or study protocol.

    Resolve relocated source paths without changing their stored hashes. A
    missing or ambiguous runtime raises ValueError rather than using live code.
    """
    result = resolve_recorded_path(result)
    record_path = result / "protocol.json"
    if not record_path.exists():
        record_path = result / "run.json"
    record = json.loads(record_path.read_text())
    candidates = set()
    for name in record["source_sha256"]:
        source = resolve_recorded_path(name)
        for parent in source.parents:
            if parent.name == "runtime" and (parent.parent / "runtime_provenance.json").is_file():
                candidates.add(parent)
                break
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one recorded numerical runtime: {result}")
    return candidates.pop()


def activate_runtime(runtime):
    """Verify and select one frozen runtime before numerical imports.

    All Python hashes in ``runtime_provenance.json`` must match. Raise
    RuntimeError if another numerical version is already imported; restart the
    Python process or notebook kernel instead of mixing model implementations.
    Return the absolute runtime directory. No snapshot files are rewritten.
    """
    global _active_runtime, _core_package
    runtime = resolve_recorded_path(runtime)
    package = runtime_package_directory(runtime)
    if _active_runtime is not None and _active_runtime != runtime:
        raise RuntimeError("A different runtime is active; restart the Python process or notebook kernel")
    provenance = json.loads((runtime.parent / "runtime_provenance.json").read_text())
    for name, expected in provenance["runtime_sha256"].items():
        source = runtime / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen runtime source changed: {source}")
    selected = f"scripts.{package.name}.experiments.core"
    numerical_packages = tuple(f"scripts.{name}.experiments.core"
                               for name in (CURRENT_PACKAGE, LEGACY_PACKAGE))
    for name, module in tuple(sys.modules.items()):
        numerical = (name == "SpikingNN" or name.startswith("SpikingNN.")
                     or any(name == prefix or name.startswith(prefix + ".")
                            for prefix in numerical_packages))
        source = getattr(module, "__file__", None)
        if numerical and source and not Path(source).resolve().is_relative_to(runtime):
            raise RuntimeError(f"Another numerical implementation is loaded ({name}); restart the Python process or notebook kernel")
    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    # A command may already have imported the live namespace to reach this
    # helper. Extend only package search paths; never evict numerical modules.
    for name, directory in (("scripts", runtime / "scripts"),
                            (f"scripts.{package.name}", package),
                            (f"scripts.{package.name}.experiments", package / "experiments")):
        module = sys.modules.get(name)
        if module is not None and hasattr(module, "__path__"):
            paths = [str(directory), *list(module.__path__)]
            module.__path__ = list(dict.fromkeys(paths))
    _active_runtime, _core_package = runtime, selected
    return runtime


def core_module(name):
    """Import a shared-core module from the selected runtime or the live package."""
    module = importlib.import_module(f"{_core_package}.{name}")
    if _active_runtime is not None and not Path(module.__file__).resolve().is_relative_to(_active_runtime):
        raise RuntimeError(f"Numerical module resolved outside the selected runtime: {module.__file__}")
    return module
