#!/usr/bin/env python3
"""Verify current article exports and package the standalone source release.

Run after build.py and validate_article.py. Training arrays and Python sources
are never edited. The ZIP excludes local experiments, caches and old archives.
"""
from pathlib import Path
import json
import shutil
import sys
import zipfile

import numba
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = ROOT/'topubsrc'
sys.path.insert(0, str(SOURCE))
from genome import file_hash, load_solution, write_json
from model import evaluate_population
from prepare_fresh_results import EXPS, stage_for


def main():
    validation = json.loads((HERE/'validation.json').read_text())
    assert validation['status'] == 'passed'
    assert validation['pdf_sha256'] == file_hash(HERE/'Afferents2026.pdf')
    assert validation['manuscript_sha256'] == file_hash(HERE/'main.tex')
    assert file_hash(HERE/'appendix_genomes.json') == file_hash(SOURCE/'appendix_genomes.json')
    # Confirm that none of the recorded training artifacts changed during publication.
    manifest = json.loads((SOURCE/'fresh_results/manifest.json').read_text())['sha256']
    for name, expected in manifest.items():
        assert file_hash(SOURCE/'fresh_results'/name) == expected, name
    published = json.loads((SOURCE/'published_results.json').read_text())
    for row in published:
        assert file_hash(SOURCE/row['genome']) == row['published_file_sha256']
        assert file_hash(ROOT/row['source']) == row['published_file_sha256']
        assert file_hash(SOURCE/row['reference_metrics']) == row['reference_metrics_sha256']
    replay = json.loads((SOURCE/'verification.json').read_text())
    assert replay['status']=='passed' and replay['test_episodes']==1608
    assert {(r['experiment'],r['domain'],r['success']) for r in replay['results']} == {
        (r['experiment'],r['domain'],r['test_success']) for r in published}
    # Independent dynamics for each appendix genome on its selection bank.
    bundle = json.loads((HERE/'appendix_genomes.json').read_text())
    numba.set_num_threads(min(8, numba.config.NUMBA_NUM_THREADS))
    appendix_checks = []
    for row in bundle['genomes']:
        path = SOURCE/EXPS[row['task']]/'genomes'/f"{row['id']}.json"
        assert json.loads(path.read_text()) == row['genome']
        genome, unit = load_solution(path)
        with np.load(ROOT/row['stage']/'cases.npz') as saved:
            cases=saved['validation']
        scores, metrics = evaluate_population(genome, unit[None], cases, 10.)
        expected=np.asarray(row['validation_metrics'])
        np.testing.assert_allclose(metrics[0], expected, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(scores[0], row['validation_fitness'], atol=1e-12, rtol=0)
        appendix_checks.append(dict(id=row['id'], cases=len(cases),
                                    max_absolute_difference=float(abs(metrics[0]-expected).max())))
    notebooks=[]
    for path in sorted(SOURCE.glob('*/explore.ipynb')):
        nb=json.loads(path.read_text())
        code=[c for c in nb['cells'] if c['cell_type']=='code']
        assert all(c['execution_count'] is not None for c in code)
        assert not any(o['output_type']=='error' for c in code for o in c['outputs'])
        notebooks.append(dict(path=str(path.relative_to(SOURCE)), executed_cells=len(code), sha256=file_hash(path)))
    update=dict(date='2026-11-13', status='passed', article_pdf_sha256=file_hash(HERE/'Afferents2026.pdf'),
                main_with_references_pages=validation['main_with_references_pages'],
                appendix_pages=validation['appendix_pages'], figures=validation['figures'],
                selection='Saved validation fitness only; all reported training starts from generic configuration.',
                main_genomes=7, test_episodes=1608, appendix_replays=appendix_checks,
                notebooks=notebooks, training_artifact_hashes_verified=len(manifest),
                numerical_training_sources_changed=False)
    write_json(SOURCE/'article_update.json', update)
    files=[]
    for name in ['run.py','model.py','genome.py','evolution.py','verify.py','README.md','LICENSE','.gitignore',
                 'requirements.txt','requirements-reproducible.txt','published_results.json',
                 'appendix_genomes.json','verification.json','training_verification.json','article_update.json']:
        files.append(SOURCE/name)
    for folder in EXPS.values():
        files.extend(SOURCE/folder/name for name in ['experiment.py','genome_cfg.json','training.json','profiles.json','explore.ipynb'])
        files.extend(sorted((SOURCE/folder/'genomes').glob('*.json')))
        files.extend(sorted((SOURCE/folder/'reference').glob('*.npz')))
    files.extend(p for p in sorted((SOURCE/'fresh_results').rglob('*')) if p.is_file())
    destination=ROOT/'topubsrc.zip'
    backup=Path('/tmp/topubsrc_before_article_update_20261113.zip')
    if destination.exists() and not backup.exists():shutil.copy2(destination,backup)
    temporary=ROOT/'topubsrc.next.zip'
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(files):z.write(path,str(Path('topubsrc')/path.relative_to(SOURCE)))
    with zipfile.ZipFile(temporary) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(files)==len(set(z.namelist()))
        import hashlib
        for path in files:
            name=str(Path('topubsrc')/path.relative_to(SOURCE))
            assert hashlib.sha256(z.read(name)).hexdigest()==file_hash(path),name
    temporary.replace(destination)
    digest=file_hash(destination)
    (ROOT/'topubsrc.zip.sha256').write_text(digest+'  topubsrc.zip\n')
    print(json.dumps(dict(status='passed',main_test_episodes=1608,
                          appendix_validation_episodes=sum(r['cases'] for r in appendix_checks),
                          notebooks=len(notebooks),zip_entries=len(files),zip_bytes=destination.stat().st_size,
                          zip_sha256=digest),indent=2))


if __name__=='__main__':
    main()
