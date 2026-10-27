"""Verify relocated paths and strict selection of immutable numerical snapshots."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.AfferentBalance2026.tools import runtime_support as runtime


def test_renamed_outer_directory_preserves_nested_historical_namespace(tmp_path, monkeypatch):
    """Remap the experiment root while preserving package names inside snapshots."""
    base = tmp_path / "scripts/AfferentBalance2026"
    monkeypatch.setattr(runtime, "BASE", base)
    old = tmp_path / "scripts/SteppingReflexGA/report/results/run/runtime/scripts/SteppingReflexGA/experiments/core/config.py"
    expected = base / "report/results/run/runtime/scripts/SteppingReflexGA/experiments/core/config.py"
    assert runtime.resolve_recorded_path(old) == expected


def test_relative_pointer_is_anchored_at_its_own_directory(tmp_path):
    """Resolve a portable current-series pointer independently of the working directory."""
    root = tmp_path / "series"
    root.mkdir()
    (root / "manifest.json").write_text('{}')
    pointer = tmp_path / "current.json"
    pointer.write_text(json.dumps({'root': 'series'}))
    assert runtime.current_results(pointer) == root


@pytest.mark.parametrize('package', ['SteppingReflexGA', 'AfferentBalance2026'])
def test_verified_snapshot_imports_from_its_own_namespace(tmp_path, package):
    """Load both old and renamed package layouts in a fresh Python process."""
    root = tmp_path / "runtime"
    core = root / "scripts" / package / "experiments/core"
    core.mkdir(parents=True)
    for directory in (root / "scripts", core.parent.parent, core.parent, core):
        (directory / "__init__.py").write_text('')
    source = core / "config.py"
    source.write_text('VALUE = 7\n')
    provenance = {'runtime_sha256': {str(source.relative_to(root)):
                                    hashlib.sha256(source.read_bytes()).hexdigest()}}
    (tmp_path / "runtime_provenance.json").write_text(json.dumps(provenance))
    script = (
        'from scripts.AfferentBalance2026.tools.runtime_support import activate_runtime, core_module\n'
        'import sys\n'
        'activate_runtime(sys.argv[1])\n'
        'module = core_module("config")\n'
        'assert module.VALUE == 7\n'
        'print(module.__file__)\n'
    )
    result = subprocess.run([sys.executable, '-c', script, str(root)],
                            cwd=runtime.REPO, check=True, capture_output=True, text=True)
    assert str(source) in result.stdout


def test_modified_snapshot_is_rejected_before_import(tmp_path):
    """Reject a changed source instead of silently updating its provenance hash."""
    root = tmp_path / "runtime"
    core = root / "scripts/SteppingReflexGA/experiments/core"
    core.mkdir(parents=True)
    source = core / "config.py"
    source.write_text('VALUE = 8\n')
    provenance = {'runtime_sha256': {str(source.relative_to(root)): '0' * 64}}
    (tmp_path / "runtime_provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match='Frozen runtime source changed'):
        runtime.activate_runtime(root)


@pytest.mark.parametrize('record_name', ['protocol.json', 'run.json'])
def test_recorded_runtime_relocates_sources_without_changing_metadata(tmp_path, monkeypatch, record_name):
    """Locate a study's original runtime after renaming its outer experiment path."""
    base = tmp_path / "scripts/AfferentBalance2026"
    monkeypatch.setattr(runtime, "BASE", base)
    root = base / "report/results/study"
    snapshot = root / "factor_1/runtime"
    snapshot.mkdir(parents=True)
    (snapshot.parent / "runtime_provenance.json").write_text('{}')
    old_source = tmp_path / "scripts/SteppingReflexGA/report/results/study/factor_1/runtime/scripts/SteppingReflexGA/experiments/core/dynamics.py"
    record = root / record_name
    original = json.dumps({'source_sha256': {str(old_source): 'saved-source-hash'}})
    record.write_text(original)
    assert runtime.recorded_runtime(root) == snapshot
    assert record.read_text() == original


def test_recorded_runtime_does_not_fall_back_to_live_sources(tmp_path):
    """Reject missing frozen-source provenance instead of silently using live code."""
    (tmp_path / 'protocol.json').write_text(json.dumps({'source_sha256': {}}))
    with pytest.raises(ValueError, match='exactly one recorded numerical runtime'):
        runtime.recorded_runtime(tmp_path)


def test_cross_namespace_numerical_mix_is_rejected(tmp_path):
    """Reject a legacy snapshot when a live renamed core is already imported."""
    snapshot = tmp_path / "runtime"
    core = snapshot / "scripts/SteppingReflexGA/experiments/core"
    core.mkdir(parents=True)
    (tmp_path / 'runtime_provenance.json').write_text('{"runtime_sha256": {}}')
    script = (
        'from scripts.AfferentBalance2026.tools.runtime_support import activate_runtime\n'
        'from scripts.AfferentBalance2026.experiments.core import genome\n'
        'import sys\n'
        'try:\n'
        '    activate_runtime(sys.argv[1])\n'
        'except RuntimeError as error:\n'
        '    assert "Another numerical implementation" in str(error)\n'
        'else:\n'
        '    raise AssertionError("Mixed numerical implementations accepted")\n'
    )
    subprocess.run([sys.executable, '-c', script, str(snapshot)],
                   cwd=runtime.REPO, check=True, capture_output=True, text=True)
