"""Load current C/D genome codecs and resolve explicit notebook genome choices.

The simulation and trained JSON files are never rewritten here. A long-lived
Jupyter kernel can retain codecs imported before mirror constraints were added;
refresh their Python dependencies before binding notebook classes and functions.
"""
import hashlib
import importlib
from pathlib import Path
import sys


_PREFIX = 'scripts.AfferentBalance2026.experiments.'
_MODULES = (
    'core.genome', 'core.config', 'core.artifacts', 'core.run', 'C_setpoint_balance.polynomial_study',
    'C_setpoint_balance.joint_study', 'C_setpoint_balance.run_experiment',
    'D_trajectory.controller', 'D_trajectory.run_experiment',
    'D_trajectory', 'visualization',
)
# Notebook cells reload this helper too. Preserve the successful refresh cache
# across that reload so unchanged classes retain their identity on later calls.
_SOURCE_HASHES = globals().get('_SOURCE_HASHES', {})
GENOME_VARIANTS = {
    'fast_muscle': 'best_genome_fast_muscle.json',
    'slow_muscle': 'best_genome_slow_muscle.json',
    'slow_muscle_mirrow': 'best_genome_slow_muscle_mirrow.json',
    'current': 'results/best_genome.json',
}


def refresh_model_modules():
    """Refresh dependent codecs together when first called or source files change.

    Call before importing ``GenomeConfig``, ``AngleController`` or other notebook
    aliases. Subsequent calls with unchanged sources preserve class identities.
    Only codec/training/visualization wrappers are refreshed; the numerical
    integrators and state of existing controllers are not reset or reloaded.
    """
    # Inspect source files without importing dependants: a cached old genome
    # may lack symbols required even to import the current visualization module.
    root = Path(__file__).resolve().parent
    sources = {__name__: Path(__file__).resolve()}
    for name in _MODULES:
        path = root.joinpath(*name.split('.'))
        sources[_PREFIX + name] = (path / '__init__.py' if path.is_dir()
                                   else path.with_suffix('.py'))
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in sources.items()}
    if hashes != _SOURCE_HASHES:
        # If any reload fails, leave the cache empty so a retry refreshes every
        # dependency instead of accepting a partially updated module graph.
        _SOURCE_HASHES.clear()
        importlib.invalidate_caches()
        for name in _MODULES:
            fullname = _PREFIX + name
            if fullname in sys.modules:
                importlib.reload(sys.modules[fullname])
            else:
                importlib.import_module(fullname)
        _SOURCE_HASHES.update(hashes)


def resolve_genome_path(experiment, path=None, *, variant='slow_muscle_mirrow'):
    """Resolve a named variant or an explicit JSON/population path independently of cwd.

    Parameters
    ----------
    experiment : path-like
        Absolute experiment directory, normally C_setpoint_balance or D_trajectory.
    path : path-like, optional
        Explicit override. Bare filenames are anchored to the experiment folder;
        longer relative paths are searched from cwd, the experiment and the repo.
    variant : str
        One of GENOME_VARIANTS; used only when path is None. Missing selections
        raise an error instead of silently falling back to another trained genome.
    """
    experiment = Path(experiment).expanduser().resolve()
    if path is None:
        if variant not in GENOME_VARIANTS:
            raise ValueError(f'Unknown genome variant {variant!r}; choose {tuple(GENOME_VARIANTS)}')
        choices = [experiment / GENOME_VARIANTS[variant]]
    else:
        source = Path(path).expanduser()
        if source.is_absolute():
            choices = [source]
        elif len(source.parts) == 1:
            choices = [experiment / source]
        else:
            repo = Path(__file__).resolve().parents[3]
            choices = [Path.cwd() / source, experiment / source, repo / source]
    for candidate in choices:
        if candidate.is_file():
            return candidate.resolve()
    searched = ', '.join(str(candidate.resolve()) for candidate in choices)
    raise FileNotFoundError(f'Selected genome was not found: {searched}')


# Correct spelling is preferred; the published historical filename stays intact.
GENOME_VARIANTS['slow_muscle_mirror'] = GENOME_VARIANTS['slow_muscle_mirrow']
_RUNTIME_HASHES = globals().get('_RUNTIME_HASHES', {})


def check_runtime():
    """Detect changed numerical sources; never auto-reload a running simulator."""
    from .core.config import project_root
    root = project_root()
    paths = [root/'experiments/simulation.py', root/'experiments/core/genome.py',
             root/'experiments/D_trajectory/trajectory.py', root/'experiments/D_trajectory/controller.py',
             root/'experiments/C_setpoint_balance/joint_study.py']
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for name, module in tuple(sys.modules.items()):
        source = getattr(module, '__file__', None)
        if name.startswith(_PREFIX) and source and not Path(source).resolve().is_relative_to(root):
            raise RuntimeError('An archived experiment runtime is already imported. Restart the kernel before current-model replay.')
    if _RUNTIME_HASHES and hashes != _RUNTIME_HASHES:
        raise RuntimeError('Numerical/codec sources changed. Restart the Jupyter kernel before loading a solution; '
                           'existing state cannot be mixed with a new or archived runtime.')
    from .core import genome
    if not hasattr(genome, 'MIRROR_INDICES'):
        raise RuntimeError('Cached pre-mirror genome module. Restart the Jupyter kernel.')
    _RUNTIME_HASHES.update(hashes)


def select_solution(experiment, solution=None, *, variant='current', rank=1, smoke=False, initial=False):
    """Load one common Solution, or an explicitly untrained in-memory smoke fixture.

    Smoke mode never relies on ignored training results and never runs a GA.
    Normal missing-file errors do not fall back to a different genome.
    """
    from .core.artifacts import load_solution, genome_payload, make_codec
    from .core.genome import GenomeConfig
    import numpy as np
    check_runtime()
    experiment = Path(experiment).resolve()
    if smoke or initial:
        cfg = GenomeConfig(experiment/'genome_cfg.json')
        codec = make_codec(cfg, degree=cfg.data.get('setpoint', {}).get('degree', 3))
        if cfg.data['experiment'][0] in 'AB':
            unit = cfg.initial()
        else:
            unit = codec.encode(cfg.initial_physical, np.zeros((2, getattr(codec, 'width', 4))))
        selected = load_solution(genome_payload(codec, unit, selection='untrained smoke fixture'))
    else:
        selected = load_solution(resolve_genome_path(experiment, solution, variant=variant), rank=rank)
    print(selected.describe())
    return selected


def controller_input(solution):
    """Return the selected path or an in-memory portable document for a smoke UI."""
    from .core.artifacts import genome_payload
    return solution.path if solution.path is not None else genome_payload(solution.codec, solution.unit, solution.current_tau_s)
