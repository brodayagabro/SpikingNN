#!/usr/bin/env python3
"""Select three distinct genomes per task from frozen validation checkpoints.

No training, simulation or test-based ranking. The candidate pool is restricted
to the corrected-tension branches catalogued in the manuscript, with the same
validation bank within each task. Outputs include portable full-precision
genomes, selection provenance and typeset physical-parameter tables.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT/'topubsrc'))
from genome import GenomeConfig, BalanceGenome, JointGenome, file_hash
from prepare_fresh_results import stage_for

TASKS = 'ABC'
INPUTS = {}
DOMAIN_NAMES = {'fast': 'Fast', 'slow': 'Slow', 'mirror': 'Slow mirror'}


def digest(path):
    return file_hash(path)


def record(path):
    path = Path(path)
    INPUTS[str(path.relative_to(ROOT))] = digest(path)
    return path


def read_json(path):
    return json.loads(record(path).read_text())


def codec_for(task, genome):
    if task in 'AB':
        return BalanceGenome(GenomeConfig(genome['configuration']))
    if task == 'C':
        return JointGenome(GenomeConfig(genome['configuration']), genome['command']['kind'], genome['command']['degree'])
    raise ValueError(f'Task outside the article scope: {task}')


def decode(task, codec, unit):
    value = codec.decode(unit)
    return np.r_[value[0], value[1].ravel()]


def bank_for(task, stage):
    with np.load(record(stage/'cases.npz'), allow_pickle=False) as saved:
        bank = saved['validation']
    return bank.tobytes(), len(bank)


def collect():
    """Return sorted, deduplicated candidates and an audit of eligible stages."""
    INPUTS.clear()
    manifest = read_json(HERE/'article_manifest.json')
    assert manifest['article_tasks'] == list(TASKS)
    for name in ('topubsrc/genome.py', 'topubsrc/C_setpoint_balance/experiment.py'):
        record(ROOT/name)
    reference = {}
    for task in TASKS:
        run = next(r for r in manifest['selected_results'] if r['task'] == task and r['group'] == 'fast')
        reference[task] = bank_for(task, stage_for(ROOT/run['run']))
    candidates = {task: [] for task in TASKS}
    audit = []
    for run in manifest['run_catalog']:
        task = run['task']
        assert task in TASKS
        root = ROOT/run['run']
        for stage in sorted({p.parent for p in root.rglob('validation_*.npz')}):
            template = read_json(stage/'best_genome.json')
            stage_name = str(stage.relative_to(ROOT))
            if task == 'C' and template['command']['kind'] != 'polynomial':
                audit.append(dict(stage=stage_name, included=False, reason='Table command; different validation targets'))
                continue
            bank, count = bank_for(task, stage)
            if (bank, count) != reference[task]:
                audit.append(dict(stage=stage_name, included=False, reason='Different validation bank'))
                continue
            result = read_json(root/'provenance.json')
            settings = read_json(root/'training.json')
            assert result['model'] == manifest['model']
            codec = codec_for(task, template)
            np.testing.assert_allclose(decode(task, codec, template['normalized']), template['physical'], atol=1e-12, rtol=0)
            assert np.array_equal(np.asarray(template['physical'])[[18, 25]], [.37, .37])
            before = len(candidates[task])
            sources = [(p, int(p.stem.split('_')[-1])) for p in sorted(stage.glob('validation_*.npz'))]
            sources.insert(0, (stage/'baseline_validation.npz', 0))
            for path, generation in sources:
                if generation == 0:
                    initial = read_json(stage/'initial_genome.json')
                    units = np.asarray(initial['normalized'])[None, :]
                    np.testing.assert_allclose(decode(task, codec, units[0]), initial['physical'], atol=1e-12, rtol=0)
                with np.load(record(path), allow_pickle=False) as saved:
                    if generation:
                        units = saved['normalized']
                    scores = saved['scores']
                    metrics = saved['metrics'].reshape(len(scores), count, 9)
                assert len(units) == len(scores) and np.isfinite(metrics).all()
                episode_scores = metrics[:, :, 0]
                computed = (.5*episode_scores.mean(axis=1)
                            + .3*np.sort(episode_scores, axis=1)[:, :int(np.ceil(count/4))].mean(axis=1)
                            + .2*episode_scores.min(axis=1))
                np.testing.assert_allclose(scores, computed, atol=1e-12, rtol=0)
                for index, (unit, score, measured) in enumerate(zip(units, scores, metrics)):
                    physical = decode(task, codec, unit)
                    candidates[task].append(dict(
                        task=task, domain=run['group'], run=run['run'], stage=stage_name,
                        source=str(path.relative_to(ROOT)), generation=generation, individual=index,
                        seed=settings['seed'], validation_fitness=float(score), validation_cases=count,
                        validation_success=int(measured[:, 5].sum()),
                        normalized=unit.tolist(), physical=physical.tolist(),
                        template=template, metrics=measured.tolist()))
            audit.append(dict(stage=stage_name, included=True, candidates=len(candidates[task])-before))
    selected, counts = [], {}
    for task, rows in candidates.items():
        rows.sort(key=lambda r: (-r['validation_fitness'], r['stage'], r['generation'], r['individual']))
        unique, seen = [], set()
        for row in rows:
            # Numerical identity, not a diversity objective or a new fitness term.
            key = tuple(np.round(row['physical'], 12))
            if key in seen:
                continue
            seen.add(key)
            unique.append(row)
        assert len(unique) >= 3
        counts[task] = dict(evaluated_entries=len(rows), distinct_genomes=len(unique), validation_cases=reference[task][1],
                            validation_bank_sha256=hashlib.sha256(reference[task][0]).hexdigest())
        for rank, row in enumerate(unique[:3], 1):
            row['id'] = f'{task}{rank}'
            row['rank'] = rank
            selected.append(row)
    return selected, counts, audit


def portable(row):
    genome = deepcopy(row['template'])
    genome['physical'] = row['physical']
    genome['normalized'] = row['normalized']
    if 'parameters' in genome:
        for field, value in zip(genome['parameters'], row['physical']):
            field['value'] = value
    genome['metadata'] = dict(selection='Appendix: top three distinct genomes by saved validation fitness',
                              model='SpikingNN.Afferents/v3-tensile', generation=row['generation'],
                              individual=row['individual'], seed=row['seed'], source=row['source'])
    return genome


def parameter_rows(task):
    rows = [(f'${key}_i$', [2*i, 2*i+1]) for i, key in enumerate('abcd')]
    rows += [(rf'$W_{{{i}j}}$', list(range(8+2*i, 10+2*i))) for i in range(2)]
    rows += [(rf'$\tau_{{{i}j}}$', list(range(12+2*i, 14+2*i))) for i in range(2)]
    rows += [(f'${key}$', [16+i, 23+i]) for i, key in enumerate(('w_m', 'N_m', 'A_m', r'\tau_{c,m}', r'\tau_{1,m}', 'n_m', 'k_m'))]
    rows += [(rf'$Q_{{i,{channel}}}$', [30+i, 36+i]) for i, channel in enumerate(('Ia_F', 'II_F', 'Ib_F', 'Ia_E', 'II_E', 'Ib_E'))]
    rows += [(r'$L_{{\rm th},m}$', [42, 43]), (r'$F_{{\rm th},m}$', [44, 45]),
             (r'$(a_1,a_2)$', [46, 47]), (r'$P_{Fi}$', [48, 49]), (r'$P_{Ei}$', [50, 51])]
    if task == 'C':
        width = 3
        rows += [(rf'$c_{{i{k}}}$', [52+k, 52+width+k]) for k in range(width)]
    assert sorted(i for _, indices in rows for i in indices) == list(range(52 if task in 'AB' else 58))
    return rows


def tex_for(selected, counts):
    lines = [r'% Generated by build_appendices.py; edit the generator, not the values.',
             r'\appendix', r'\clearpage']
    for task in TASKS:
        rows = [r for r in selected if r['task'] == task]
        lines += [rf'\section{{Three best recorded genomes: experiment {task}}}', rf'\label{{app:genomes{task}}}']
        if task == 'A':
            lines += [r'Three distinct genomes per task are ranked by saved validation fitness on identical within-task banks. Candidates come from the initial and checkpoint populations of the 43 fresh run records described in Sect.~\ref{sec:search}; C table commands are excluded because their validation targets differ. Test scores are not used. Duplicates at 12 decimal places are removed. Domains and budgets differ; neighbouring candidates are not independent replications. Ranking uses saved validation metrics only.\par']
        lines += [f"For {task}, the pool contains {counts[task]['evaluated_entries']:,} evaluated entries and {counts[task]['distinct_genomes']:,} distinct genomes, each scored on {counts[task]['validation_cases']} validation episodes.",
                  r'Physical parameters are rounded to six significant figures. The accompanying \texttt{appendix\_genomes.json} retains full precision, normalized coordinates, configurations and source paths.\par']
        lines += [r'\begin{center}\fontsize{8.5}{10}\selectfont', r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{@{}lrrr@{}}', r'\toprule',
                  ' & '+' & '.join(r['id'] for r in rows)+r'\\', r'\midrule',
                  'Domain & '+' & '.join(DOMAIN_NAMES[r['domain']] for r in rows)+r'\\',
                  'Validation fitness & '+' & '.join(f"{r['validation_fitness']:.9f}" for r in rows)+r'\\',
                  'Validation success & '+' & '.join(f"{r['validation_success']}/{r['validation_cases']}" for r in rows)+r'\\',
                  'Seed / generation / index & '+' & '.join(f"{r['seed']} / {r['generation']} / {r['individual']}" for r in rows)+r'\\', r'\midrule']
        for label, indices in parameter_rows(task):
            lines.append(label+' & '+' & '.join(', '.join(f'{r["physical"][i]:.6g}' for i in indices) for r in rows)+r'\\')
        lines += [r'\bottomrule', r'\end{tabular}\end{center}',
                  r'Pairs follow $i,j=0,1$ or $m=F,E$. For $Q$, they list neurons receiving the named channel; for $P$, neuronal inputs to the named muscle. Times are in ms, lengths in m and forces in N; remaining units follow the model.']
        if task == 'C':
            lines += [r'Command coefficients use $I_i^{\rm app}=\sum_k c_{ik}\xi^k$, $\xi=\theta^*/45^\circ$, in model current units.']
        else:
            lines += [r'Applied command currents are zero.']
        sources = {}
        for row in rows:
            sources.setdefault((row['run'], row['stage']), []).append(row['id'])
        lines += [r'Indices are zero-based. Sources: '+
                  '; '.join(', '.join(ids)+': '+r'\texttt{'+Path(run).name.replace('_', r'\_')+'} / '+Path(stage).name
                            for (run, stage), ids in sources.items())+'.', r'\clearpage']
    return '\n'.join(lines)+'\n'


def main():
    selected, counts, audit = collect()
    tex = tex_for(selected, counts)
    bundle = dict(selection='Top three distinct genomes per task by saved validation fitness on identical within-task banks; no test selection.',
                  generator_sha256=digest(Path(__file__)), tex_sha256=hashlib.sha256(tex.encode()).hexdigest(),
                  pool=counts, stages=audit, input_sha256=INPUTS,
                  genomes=[dict(**{k:v for k,v in r.items() if k not in ('template','physical','normalized','metrics')},
                                validation_metrics=r['metrics'], genome=portable(r)) for r in selected])
    (HERE/'appendix_genomes.json').write_text(json.dumps(bundle, indent=2, allow_nan=False)+'\n')
    (HERE/'appendix_genomes.tex').write_text(tex)
    import shutil
    from prepare_fresh_results import EXPS, SOURCE
    shutil.copy2(HERE/'appendix_genomes.json', SOURCE/'appendix_genomes.json')
    for row in selected:
        destination = SOURCE/EXPS[row['task']]/'genomes'/f"{row['id']}.json"
        destination.write_text(json.dumps(portable(row), indent=2, allow_nan=False)+'\n')
    published=json.loads((SOURCE/'published_results.json').read_text())
    for item in published:
        if item['domain'] != 'fast':continue
        item['appendix_genomes']=[dict(id=r['id'], rank=r['rank'], generation=r['generation'], individual=r['individual'], seed=r['seed'], validation_fitness=r['validation_fitness'], validation_cases=r['validation_cases'], validation_success=r['validation_success'], source=r['source'], genome=f"{EXPS[r['task']]}/genomes/{r['id']}.json", published_file_sha256=digest(SOURCE/EXPS[r['task']]/'genomes'/f"{r['id']}.json")) for r in selected if r['task']==item['experiment']]
    (SOURCE/'published_results.json').write_text(json.dumps(published,indent=2)+'\n')
    for row in selected:
        print(row['id'], row['domain'], Path(row['run']).name, row['generation'], row['individual'], f"{row['validation_fitness']:.9f}")
    print('appendix_genomes.json / appendix_genomes.tex')


if __name__ == '__main__':
    main()
