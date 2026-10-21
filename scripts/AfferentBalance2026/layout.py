"""Paths shared by experiment scripts; no simulation or search logic."""
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile

if __package__:
    from .runtime_support import resolve_recorded_path
else:
    from runtime_support import resolve_recorded_path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
EXPERIMENTS = ROOT / "experiments"


def image_path(path, create=True):
    """Route experiment report/results images to the matching img directory.

    Explicit output directories outside this layout retain their usual behavior.
    """
    path = Path(path).resolve()
    try:
        parts = path.relative_to(EXPERIMENTS).parts
    except ValueError:
        parts = ()
    if len(parts) >= 4 and parts[1:3] == ("report", "results"):
        path = EXPERIMENTS / parts[0] / "img" / Path(*parts[3:])
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def image_link(directory, filename):
    """Return a relative link to an image after applying the experiment image layout."""
    return Path(os.path.relpath(
        image_path(Path(directory) / filename, create=False), directory)).as_posix()


def report_image_links(text, directory):
    """Keep generated Markdown links consistent with the separate img tree."""
    def replace(match):
        """Rewrite a local image target while leaving URLs and data links unchanged."""
        target = match.group(1)
        if "://" in target or Path(target).suffix.lower() not in (".png", ".pdf", ".svg"):
            return match.group(0)
        return "](" + image_link(directory, target) + ")"
    return re.sub(r"\]\(([^)\n]+)\)", replace, text)


def recorded_source(path):
    """Resolve a historical filename without modifying archived run metadata."""
    path = Path(path)
    path = resolve_recorded_path(path, relative_to=REPO)
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError:
        return path
    previous = ROOT / "archive/pre_ABC_20261020" / relative
    if relative.startswith("experiments/") and previous.is_file():
        return previous
    migration_path = ROOT / "archive/core_migration.json"
    migration = json.loads(migration_path.read_text()) if migration_path.exists() else {"files": {}}
    if relative in migration["files"]:
        return ROOT / migration["files"][relative]["path"]
    manifest_path = ROOT / "archive/layout_migration.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
    item = manifest["files"].get(relative)
    if item:
        destination = item["path"]
        destination = migration["files"].get(destination, {}).get("path", destination)
        return ROOT / destination
    return path


def verify_recorded_source(path, expected):
    """Verify live data or explicitly archived historical source, never rehash it.

    Training metadata keeps the hashes from the actual training run. Relocated
    source files may have changed imports/output paths; those old hashes are
    checked against the source snapshot, not replaced by current-code hashes.
    """
    path = Path(path)
    path = resolve_recorded_path(path, relative_to=REPO)
    current = recorded_source(path)
    if current.is_file() and hashlib.sha256(current.read_bytes()).hexdigest() == expected:
        return "archived" if ROOT / "archive" in current.parents else "current"
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError:
        raise AssertionError(f"Source changed: {path}") from None
    snapshot = ROOT / "archive/before_structure_20261008.zip"
    if not snapshot.is_file():
        raise AssertionError(f"Source missing or changed, historical snapshot unavailable: {path}")
    with zipfile.ZipFile(snapshot) as archive:
        try:
            original = archive.read(relative)
        except KeyError:
            raise AssertionError(f"Source missing or changed: {path}") from None
    assert hashlib.sha256(original).hexdigest() == expected, path
    return "archived"
