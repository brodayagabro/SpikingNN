#!/usr/bin/env python3
"""Validate article provenance, saved metrics, references and the compiled PDF.

This is an artifact check, not another model-training or simulation test suite.
It only reads the existing experiments and writes validation.json locally.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
AUTHOR_SHA256 = 'cbe3b8edd2b3ed06b6461df049ed5dc09c7f7c3966d73d9f02929c28f03ffcc8'
MAX_ARTICLE_PAGES = 20  # Includes references; genome appendices are additional by agreement.
LITERATURE_EXTENSION_KEYS = {
    'peterka2002', 'loram2005', 'mileusnic2006spindle', 'mileusnic2006gto',
    'dimitriou2014', 'blum2020', 'fink2014', 'akay2014', 'maris2025', 'goulding2025',
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_restored_evidence(tex, manifest):
    """Check every search input, validation-only selection and exported coefficients."""
    from prepare_fresh_results import SOURCE, FRESH, selected_runs, stage_for
    evidence = {}
    assert len(manifest['run_catalog']) == 43
    for run in manifest['run_catalog']:
        path = ROOT/run['run']
        for name in ('training.json','result.json','provenance.json'):
            evidence[str((path/name).relative_to(ROOT))] = digest(path/name)
        assert json.loads((path/'training.json').read_text()) == run['options']
        assert json.loads((path/'result.json').read_text()) == run['result']
        provenance=json.loads((path/'provenance.json').read_text())
        assert provenance['initialization']=='configuration' and not provenance['archive_genomes_used']
        assert not provenance['smoke_test']
        for name,expected in provenance['code_sha256'].items():
            assert digest(SOURCE/name)==expected, f'Training source changed: {name}'
    for task in 'ABC':
        rows=json.loads((FRESH/f'expanded_{task}.json').read_text())
        selection=json.loads((FRESH/f'expanded_{task}_selection.json').read_text())
        assert selection['selected']['validation_fitness']==max(r['validation_fitness'] for r in rows)
        assert not selection['test_used_for_selection']
    law=(HERE/'current_laws.tex').read_text()
    g=json.loads((selected_runs()['C','fast']/'best_genome.json').read_text())
    for i in range(2):
        coefficients=re.search(rf'I_{i}\^C&=([^\n]+)',law).group(1)
        displayed=[float(x) for x in re.findall(r'[+-]?\d+\.\d+',coefficients)]
        np.testing.assert_allclose(displayed,g['physical'][52+3*i:55+3*i],rtol=0,atol=.0005)
    for row in manifest['selected_results']:
        run=ROOT/row['run']
        final=stage_for(run)
        result=json.loads((run/'result.json').read_text())
        for bank,metrics_path in [('training',run/'selected_training.npz'),('validation',final/'selected_validation.npz'),('test',run/'test_metrics.npz')]:
            with np.load(metrics_path) as z: m=z['metrics'].reshape(-1,9)
            n=len(m)
            score=.5*m[:,0].mean()+.3*np.sort(m[:,0])[:int(np.ceil(n/4))].mean()+.2*m[:,0].min()
            np.testing.assert_allclose(result[bank]['fitness'],score,atol=1e-12,rtol=0)
            assert result[bank]['cases']==n and result[bank]['success']==int(m[:,5].sum())
    return evidence


def validate_genome_appendices():
    """Recompute the archive ranking and check every exported and typeset gene."""
    import build_appendices as appendix
    bundle = json.loads((HERE/'appendix_genomes.json').read_text())
    assert digest(HERE/'build_appendices.py') == bundle['generator_sha256']
    assert digest(HERE/'appendix_genomes.tex') == bundle['tex_sha256']
    for name, expected in bundle['input_sha256'].items():
        assert digest(ROOT/name) == expected, f'Changed appendix input: {name}'
    selected, counts, audit = appendix.collect()
    assert bundle['pool'] == counts and bundle['stages'] == audit
    assert bundle['input_sha256'] == appendix.INPUTS
    assert len(bundle['genomes']) == len(selected) == 9
    for saved, row in zip(bundle['genomes'], selected):
        expected = dict(**{k:v for k,v in row.items() if k not in ('template','physical','normalized','metrics')},
                        validation_metrics=row['metrics'], genome=appendix.portable(row))
        assert saved == expected, f'Incorrect appendix genome or ranking: {row["id"]}'
        codec = appendix.codec_for(row['task'], saved['genome'])
        decoded = appendix.decode(row['task'], codec, saved['genome']['normalized'])
        np.testing.assert_allclose(decoded, saved['genome']['physical'], atol=1e-12, rtol=0)
    assert (HERE/'appendix_genomes.tex').read_text() == appendix.tex_for(selected, counts)
    return dict(genomes=9, per_task=3, ranking_recomputed=True,
                parameters_verified=sum(len(r['physical']) for r in selected),
                validation_entries=sum(r['evaluated_entries'] for r in counts.values()),
                inputs_hashed=len(bundle['input_sha256']),
                bundle_sha256=digest(HERE/'appendix_genomes.json'),
                tex_sha256=digest(HERE/'appendix_genomes.tex'))


def main():
    manifest = json.loads((HERE/'article_manifest.json').read_text())
    assert digest(HERE/'authors.tex') == AUTHOR_SHA256, 'Author block changed'
    assert digest(HERE/'build_figures.py') == manifest['generator_sha256'], 'Rebuild figures after generator changes'
    for name, expected in manifest['input_sha256'].items():
        assert digest(ROOT/name) == expected, f'Changed input: {name}'
    for name, expected in manifest['output_sha256'].items():
        assert digest(HERE/name) == expected, f'Changed figure: {name}'
    tex = '\n'.join((HERE/name).read_text() for name in ('main.tex','appendix_genomes.tex','protocols_table.tex','current_laws.tex','results_table.tex'))
    figures = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', tex)
    assert len(figures) == len(set(figures)) == manifest['figure_count'] == 5
    assert manifest['figure_tasks'] == manifest['article_tasks'] == ['A', 'B', 'C']
    assert {r['task'] for r in manifest['selected_results']} == set('ABC')
    assert {r['task'] for r in manifest['run_catalog']} == set('ABC')
    assert len(manifest['selected_results']) == 7
    assert sorted(figures) == sorted(p.name for p in (HERE/'figures').glob('*.pdf'))
    citations = set(','.join(re.findall(r'\\cite\{([^}]+)\}', tex)).split(','))
    keys = set(re.findall(r'@\w+\{([^,]+),', (HERE/'references.bib').read_text()))
    assert citations <= keys, f'Missing citations: {citations-keys}'
    compiled_keys = re.findall(r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}',
                               (HERE/'build/Afferents2026.bbl').read_text())
    assert len(compiled_keys) == len(set(compiled_keys)), 'Duplicate bibliography items'
    assert set(compiled_keys) == citations, 'Compiled bibliography differs from manuscript citations'
    introduction = tex.split(r'\section{Introduction and Related Work}', 1)[1].split(r'\section{Closed-Loop Model}', 1)[0]
    intro_keys = set(','.join(re.findall(r'\\cite\{([^}]+)\}', introduction)).split(','))
    assert LITERATURE_EXTENSION_KEYS <= intro_keys, 'Literature extension missing from introduction'
    assert 'marasco2023' in intro_keys, 'Afferent anatomy review is missing'
    intro_paragraphs = re.split(r'\n\s*\n', introduction.strip())
    assert len(intro_paragraphs) <= 5, 'Introduction exceeds the agreed five paragraphs'
    restored_evidence = validate_restored_evidence(tex, manifest)
    appendix_evidence = validate_genome_appendices()
    refs = set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', tex))
    labels = set(re.findall(r'\\label\{([^}]+)\}', tex))
    assert refs <= labels, f'Missing references: {refs-labels}'
    log = (HERE/'build/Afferents2026.log').read_text()
    assert not re.search(r'Overfull \\[hv]box|undefined|Rerun to get cross-references right|^!', log, re.M), 'LaTeX errors or unresolved layout/references'
    biblog = (HERE/'build/Afferents2026.blg').read_text()
    assert 'Warning--' not in biblog and 'error message' not in biblog
    text = subprocess.check_output(['pdftotext', '-layout', str(HERE/'Afferents2026.pdf'), '-'], text=True)
    assert '??' not in text
    caption_starts = ['Model and resolved internal activity', 'Learning histories',
                      'Fast-domain balance recovery', 'Two informative setpoint solutions',
                      'Performance across domains']
    for number, prefix in enumerate(caption_starts, 1):
        assert len(re.findall(rf'^\s*Fig\. {number}\. {re.escape(prefix)}', text, re.M)) == 1
    pages = [page for page in text.split('\f') if page.strip()]
    reference_page = next(i+1 for i,p in enumerate(pages) if re.search(r'^References\s*$',p,re.M))
    appendix_pages = [next(i+1 for i,p in enumerate(pages)
                          if re.search(rf'^{task}\s+Three best recorded genomes: experiment {task}',p,re.M))
                      for task in 'ABC']
    core_pages = appendix_pages[0]-1
    assert core_pages <= MAX_ARTICLE_PAGES, 'Main article including references exceeds 20 pages'
    assert appendix_pages == list(range(core_pages+1, core_pages+4)), 'Each genome appendix should occupy one page'
    assert len(pages) == core_pages+3
    # Check all published content, including text inside vector figures, without
    # confusing author initials in the bibliography with an experiment label.
    article_text = '\n'.join(pages[:reference_page-1]+pages[core_pages:])
    excluded = r'\bD(?:[123])?\b|D_trajectory|app:genomesD|trackingfitness|currentD|\b(?:tracking|sinusoidal)\b'
    for content in (tex, article_text):
        assert not re.search(excluded, content), 'Excluded experiment remains in article content'
    for filename in ('article_manifest.json', 'appendix_genomes.json', 'SEARCH_PROTOCOLS.md'):
        assert not re.search(r'D_trajectory|["\']task["\']\s*:\s*["\']D["\']', (HERE/filename).read_text())
    assert re.search(r'^References', pages[reference_page-1].strip().split('\n',1)[-1].strip())
    # Recompute displayed statistics and ensure domain comparisons use identical test banks.
    banks = {}
    for row in manifest['selected_results']:
        path = ROOT/row['run'];task=row['task']
        from prepare_fresh_results import CACHE, stage_for
        cache=CACHE/f"{task}_{row['group']}"
        if cache.exists():
            summary=json.loads((cache/'summary.json').read_text())
            assert summary['identity']['genome_sha256']==digest(path/'best_genome.json')
            assert summary['identity']['model']==manifest['model']
            for name,expected in summary['files'].items():assert digest(cache/name)==expected
        with np.load(path/'test_metrics.npz') as z:
            m=z['metrics']; score=float(z['scores'][0])
        assert np.isfinite(m).all()
        robust=.5*m[:,0].mean()+.3*np.sort(m[:,0])[:int(np.ceil(len(m)/4))].mean()+.2*m[:,0].min()
        assert np.isclose(score,robust,atol=1e-12)
        assert row['cases']==len(m) and row['success']==int(m[:,5].sum())
        with np.load(stage_for(path)/'cases.npz') as z:bank=z['test']
        complete=m[:,1]>=10-1e-5;error=m[:,7]
        serialized=bank.tobytes()
        if task in banks:assert serialized==banks[task],f'Different test banks in {task}'
        else:banks[task]=serialized
        assert row['completed']==int(complete.sum())
        np.testing.assert_allclose(row['error_all_deg'],np.degrees(error).mean(),atol=1e-12)
        np.testing.assert_allclose(row['error_completed_deg'],np.degrees(error[complete]).mean(),atol=1e-12)
    summary=dict(status='passed',article_tasks=manifest['article_tasks'],article_scope_verified=True,figures=len(figures),figure_tasks=manifest['figure_tasks'],cited_works=len(citations),
                 compiled_bibliography_items=len(compiled_keys),
                 introduction_paragraphs=len(intro_paragraphs),
                 fresh_search_records_verified=43,fresh_current_laws_verified=True,
                 restored_evidence_sha256=restored_evidence,
                 literature_extension_in_introduction=sorted(LITERATURE_EXTENSION_KEYS),
                 selected_runs=len(manifest['selected_results']),inputs_hashed=len(manifest['input_sha256']),
                 pages=len(pages),main_pages=reference_page-1,reference_pages=core_pages-reference_page+1,
                 main_with_references_pages=core_pages,appendix_pages=len(pages)-core_pages,
                 genome_appendices=appendix_evidence,
                 page_limit=MAX_ARTICLE_PAGES,page_limit_includes_references=True,page_limit_includes_appendices=False,
                 authors_preserved=True,common_test_banks_verified=True,
                 simulation_or_training_run=False,unresolved_references=0,overfull_boxes=0,
                 manuscript_sha256=digest(HERE/'main.tex'),pdf_sha256=digest(HERE/'Afferents2026.pdf'),
                 bibliography_sha256=digest(HERE/'references.bib'),
                 manifest_sha256=digest(HERE/'article_manifest.json'),
                 note='Existing svproc class-name/mathptmx warnings and an underfull float-page warning do not affect content.')
    (HERE/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
