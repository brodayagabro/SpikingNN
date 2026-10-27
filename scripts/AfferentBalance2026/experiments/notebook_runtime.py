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
    'core.genome', 'core.run', 'C_setpoint_balance.polynomial_study',
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
