"""Behavior of file routing and historical provenance after reorganization."""
from pathlib import Path
import hashlib
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from layout import ROOT, image_path, report_image_links, verify_recorded_source
import layout


@pytest.mark.parametrize("experiment", ["equilibrium", "small_nm_weights", "angle_setpoint"])
def test_figures_are_separate_but_report_links_resolve(experiment):
    base = ROOT / "experiments" / experiment
    output = base / "report/results/new/selected/genome"
    figure = image_path(output / "dynamics.png", create=False)
    assert figure == base / "img/new/selected/genome/dynamics.png"
    document = report_image_links("![Dynamics](dynamics.png)\n[Data](trace.npz)", output)
    target = document.split("](", 1)[1].split(")", 1)[0]
    assert (output / target).resolve() == figure
    assert "[Data](trace.npz)" in document


def test_explicit_external_output_is_respected(tmp_path):
    assert image_path(tmp_path / "plot.pdf") == tmp_path / "plot.pdf"


def test_provenance_does_not_accept_an_incorrect_historical_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(layout, "ROOT", tmp_path)
    with pytest.raises(AssertionError):
        verify_recorded_source(tmp_path / "genome.py", "0" * 64)


def test_previous_experiment_source_is_explicitly_identified_as_archived(tmp_path, monkeypatch):
    monkeypatch.setattr(layout, "ROOT", tmp_path)
    previous = "experiments/small_angles/source/genome.py"
    archived = tmp_path / "archive/legacy_experiments/small_angles/source/genome.py"
    archived.parent.mkdir(parents=True)
    archived.write_text("# Historical source fixture, independent of local experiment results.\n")
    (tmp_path / "archive/core_migration.json").write_text(json.dumps({"files": {
        previous: {"path": str(archived.relative_to(tmp_path))}
    }}))
    expected = hashlib.sha256(archived.read_bytes()).hexdigest()
    assert verify_recorded_source(tmp_path / previous, expected) == "archived"


def test_live_source_can_be_verified_without_historical_archives(tmp_path, monkeypatch):
    monkeypatch.setattr(layout, "ROOT", tmp_path)
    current = tmp_path / "genome.py"
    current.write_text("# Current source\n")
    assert verify_recorded_source(current, hashlib.sha256(current.read_bytes()).hexdigest()) == "current"
