"""Validate active ABC paths, English genome descriptions and documented Python APIs.

Historical reports, frozen snapshots and archived documentation remain records
of earlier protocols and are excluded from active-layout validation.
"""
import ast
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / 'experiments'
EXCLUDED = {'results', '__pycache__', '.ipynb_checkpoints'}


def documentation_files():
    """List current layout documents without scanning historical result trees."""
    roots = [ROOT/name for name in ('README.md', 'README.en.md', 'DEVELOPMENT.md', 'DEVELOPMENT.en.md')]
    return sorted(roots + [p for p in EXPERIMENTS.rglob('*.md')
                           if not EXCLUDED.intersection(p.relative_to(EXPERIMENTS).parts)])


def local_links(document):
    """Yield local Markdown link targets outside fenced code blocks."""
    text = re.sub(r'```.*?```', '', document.read_text(), flags=re.S)
    for target in re.findall(r'\]\(([^)\n]+)\)', text):
        parsed = urlsplit(target.strip('<>'))
        if not parsed.scheme and parsed.path:
            yield target, (document.parent / unquote(parsed.path)).resolve()


def main():
    """Check the three active experiments, notebook files and all field descriptions."""
    expected = {'A_balance', 'B_balance', 'C_setpoint_balance'}
    assert {p.name for p in EXPERIMENTS.iterdir() if p.is_dir() and p.name not in EXCLUDED} == expected | {'core'}
    for name in expected:
        folder = EXPERIMENTS/name
        for filename in ('run_experiment.py', 'run_sweep.py', 'genome_cfg.json',
                         'README.md', 'README.en.md', 'explore_genome.ipynb'):
            assert (folder/filename).is_file(), folder/filename
        config = json.loads((folder/'genome_cfg.json').read_text())
        assert len(config['parameters']) == 52
        for field in config['parameters']:
            for key in ('description', 'unit', 'system_parameter'):
                assert field[key] and not re.search('[А-Яа-яЁё]', field[key]), (name,field['name'],key)
        notebook = json.loads((folder/'explore_genome.ipynb').read_text())
        assert notebook['metadata']['kernelspec']['name'] == 'spikingnn'
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
                assert not cell.get('outputs'), 'Keep committed notebooks free of generated output'
    missing, broken, undocumented, pairs, links, definitions = [], [], [], 0, 0, 0
    for document in documentation_files():
        if not document.name.endswith('.en.md'):
            if not document.with_name(document.stem+'.en.md').exists():
                missing.append(str(document))
            else:
                pairs += 1
        for target, path in local_links(document):
            if not path.exists():
                broken.append(f'{document.relative_to(ROOT)} -> {target}')
            links += 1
    sources = [p for p in EXPERIMENTS.rglob('*.py')
               if not EXCLUDED.intersection(p.relative_to(EXPERIMENTS).parts)]
    for source in sources:
        tree = ast.parse(source.read_text(), filename=str(source))
        if not ast.get_docstring(tree):
            undocumented.append(str(source))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                definitions += 1
                if not ast.get_docstring(node):
                    undocumented.append(f'{source.name}:{node.name}')
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith('scripts.SteppingReflexGA')
    summary = dict(status='failed' if missing or broken or undocumented else 'passed',
                   experiments=sorted(expected), bilingual_pairs=pairs, local_links_verified=links,
                   python_modules=len(sources), documented_definitions=definitions,
                   missing_translations=missing, broken_links=broken, missing_docstrings=undocumented)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    (ROOT/'report').mkdir(exist_ok=True)
    (ROOT/'report/layout_validation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    if summary['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
