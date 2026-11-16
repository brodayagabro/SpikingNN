"""Verify the local run contract against physical values and historical GA paths."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.AfferentBalance2026.experiments.core import artifacts, run
from scripts.AfferentBalance2026.experiments.core.config import resolve_config, project_root, controller_config, content_hash
from scripts.AfferentBalance2026.experiments.core.genome import NAMES, file_hash, write_json, MIRROR_INDICES

ROOT = project_root()


def request(tmp_path, kind='A', **updates):
    """Build a fresh short protocol without depending on ignored scientific data."""
    path = ROOT/'configs/studies'/f'{kind}_smoke.json'
    override = {'storage':{'output':str(tmp_path/kind), 'figures':False, 'replay':False}}
    for key, value in updates.items():
        override.setdefault(key, {}).update(value)
    return resolve_config(path, overrides=override)


@pytest.mark.parametrize('kind', ['A', 'B', 'C'])
def test_common_ga_matches_existing_population_exactly(tmp_path, kind):
    """Refactoring storage and evaluation must not change any training population."""
    cfg = request(tmp_path, kind)
    run.execute(cfg)
    domain = controller_config(cfg)
    if kind in 'AB':
        domain.data['ga'] = {**cfg['search'], 'workers':1}
        domain.data['simulation']['save_top'] = 1
        run.train(domain, tmp_path/'old', population=3, generations=2, seed=cfg['search']['seed'],
                  mutation_probability=cfg['search']['mutation_probability'], workers=1)
    else:
        from scripts.AfferentBalance2026.experiments.C_setpoint_balance.joint_study import train_joint
        train_joint(domain, tmp_path/'old', population=3, generations=2, seed=cfg['search']['seed'],
                    probability=cfg['search']['mutation_probability'], workers=1)
    stage = 'polynomial' if kind == 'C' else 'training'
    for generation in (1, 2):
        filename = f'generation_{generation:04d}.npz'
        with np.load(tmp_path/kind/'stages'/stage/'generations'/filename) as actual, np.load(tmp_path/'old/generations'/filename) as old:
            for key in ('normalized', 'physical', 'training_scores', 'training_metrics', 'rng_state'):
                np.testing.assert_array_equal(actual[key], old[key])
    choice = artifacts.load_solution(tmp_path/kind/'solution.json')
    assert choice.metadata['selection']['generation'] == (2 if kind in 'AB' else 1)


def test_unknown_keys_and_provenance(tmp_path):
    """Typos, wrong-operation flags and layer origins are visible before computation."""
    path = tmp_path/'study.json'
    write_json(path, dict(schema='afferent_experiment_run_v1', preset='A_balance_tau50_v1', search={'population':7}))
    cfg = resolve_config(path, overrides={'search':{'population':9}})
    assert cfg['search']['population'] == 9
    assert cfg['provenance']['sources']['search.population'] == 'cli'
    assert cfg['provenance']['sources']['model.dt_ms'].startswith('preset:')
    for update, match in [({'genome':{'overrides':{'muscle.typo':{'value':1}}}}, 'muscle.typo'),
                          ({'search':{'poplation':4}}, 'poplation'),
                          ({'genome':{'command':{'degree':4}}}, 'inapplicable')]:
        with pytest.raises(ValueError, match=match):resolve_config(path, overrides=update)
    result = subprocess.run([sys.executable, str(ROOT/'run.py'), 'train', '--config', str(path), '--recording-stride', '10'], capture_output=True)
    assert result.returncode != 0 and not (tmp_path/'results').exists()


def test_fixed_free_and_transfer_domains(tmp_path):
    """Changing a domain changes encoding, never silently reuses normalized coordinates."""
    cfg = request(tmp_path)
    task = run.make_task(cfg)
    physical = task.config.initial_physical
    old = artifacts.load_solution(artifacts.genome_payload(task.codec, task.config.encode(physical)))
    changed = request(tmp_path, genome={'overrides':{'neurons.a.0':{'maximum':.2}}})
    codec = run.make_task(changed).codec
    unit = artifacts.transfer(old, codec)
    # The transferred source contains decoded, saved values rather than the
    # decimal configuration center (logarithmic round trips differ by a few ULP).
    np.testing.assert_array_equal(codec.decode(unit), old.physical)
    assert unit[0] != old.unit[0]
    fixed = request(tmp_path, genome={'overrides':{'neurons.a.0':{'mutable':False, 'value':.1}}})
    assert run.make_task(fixed).codec.size == task.codec.size-1
    for patch, pattern in [({'mutable':False}, 'requires value'), ({'minimum':0}, 'logarithmic'),
                            ({'maximum':.01}, 'bounds')]:
        with pytest.raises(ValueError, match=pattern):
            request(tmp_path, genome={'overrides':{'neurons.a.0':patch}})
    saved = tmp_path/'old.json';task.codec.save(saved, physical)
    incompatible = request(tmp_path, genome={'overrides':{'neurons.a.0':{'mutable':False, 'value':.05}}},
                           initialization={'solution':str(saved)})
    with pytest.raises(ValueError, match='neurons.a.0'):
        run.execute(incompatible)
    assert not (tmp_path/'A').exists()


def test_prepare_mirror_preserves_source_and_portability(tmp_path):
    """An explicit projection has a field audit; copying the output alone is sufficient."""
    cfg = request(tmp_path, 'C'); task = run.make_task(cfg)
    p = task.config.initial_physical.copy();p[0]=.08
    unit = task.codec.encode(p, np.arange(8).reshape(2,4))
    source = tmp_path/'source.json'
    artifacts.save_solution(source, task.codec, unit, cfg)
    before = file_hash(source)
    prepared = resolve_config(ROOT/'configs/studies/C_smoke.json', action='prepare', solution=source,
        overrides={'storage':{'output':str(tmp_path/'prepared')}, 'genome':{'mirror_symmetry':True},
                   'initialization':{'prepare':{'mirror_projection':True}}})
    result = artifacts.prepare_solution(prepared)
    selected = artifacts.load_solution(result['solution'])
    np.testing.assert_array_equal(selected.physical, selected.physical[MIRROR_INDICES])
    np.testing.assert_array_equal(selected.command[1], selected.command[0]*[1,-1,1,-1])
    assert file_hash(source) == before
    isolated = tmp_path/'copied.json';isolated.write_bytes(Path(result['solution']).read_bytes())
    again = artifacts.load_solution(isolated)
    np.testing.assert_array_equal(again.unit, selected.unit)
    assert json.loads((tmp_path/'prepared/preparation.json').read_text())['changes']


@pytest.mark.parametrize('method', ['joint', 'staged'])
def test_c_stage_budget_and_saved_generations(tmp_path, method):
    cfg = request(tmp_path, 'C', search={'method':method, 'generations':4, 'checkpoints':[1,2,4]})
    report = run.inspect_request(cfg)
    assert report['generations_total'] == 4
    run.execute(cfg)
    files = list((tmp_path/'C/stages').glob('*/generations/*.npz'))
    assert len(files) == 4
    data = artifacts.load_solution(files[0])
    assert data.command is not None
    manifest = json.loads((tmp_path/'C/manifest.json').read_text())
    assert manifest['status'] == 'completed'
    assert all('test_used_for_selection' in artifacts.load_solution(p).metadata['selection']
               for p in (tmp_path/'C/stages').glob('*/solution.json'))


def test_d_explicit_source_and_process_equivalence(tmp_path):
    c = request(tmp_path, 'C'); task = run.make_task(c)
    source = tmp_path/'seed.json'
    artifacts.save_solution(source, task.codec, task.codec.encode(task.config.initial_physical, np.zeros((2,4))), c)
    with pytest.raises(ValueError, match='initialization.sources'):
        resolve_config(ROOT/'configs/presets/D_trajectory_tau50_v1.json')
    d = request(tmp_path, 'D', initialization={'solution':str(source)})
    task = run.make_task(d);center,_,_=run._initial_candidates(d,task)
    with run.EvaluationPool(task, 1) as sequential:
        expected = sequential.evaluate(np.tile(center, (3,1)), 'training')
    with run.EvaluationPool(task, 2) as parallel:
        actual = parallel.evaluate(np.tile(center, (3,1)), 'training')
    for a,b in zip(expected,actual):np.testing.assert_array_equal(a,b)
    run.execute(d)
    selected = artifacts.load_solution(tmp_path/'D/solution.json')
    assert selected.metadata['selection']['generation'] in (0,1,2)
    assert (tmp_path/'D/stages/training/checkpoints/generation_0000/solution.json').exists()


def test_sweep_separate_horizons_and_no_restart(tmp_path):
    cfg = request(tmp_path, search={'generations':3, 'checkpoints':[1,2,3]})
    cfg['sweep']['axes'] = {'generations':[2,3]}
    cfg['action'] = 'sweep'
    run.execute_sweep(cfg)
    assert len(list((tmp_path/'A').glob('line_*/stages/training/generations/*.npz'))) == 5
    assert not (ROOT/'current.json').exists()
    direct = request(tmp_path, storage={'output':str(tmp_path/'direct')}, search={'generations':3,'checkpoints':[]})
    run.execute(direct)
    for generation in range(1,4):
        file=f'generation_{generation:04d}.npz'
        with np.load(tmp_path/'A/line_0001/stages/training/generations'/file) as sweep, np.load(tmp_path/'direct/stages/training/generations'/file) as single:
            np.testing.assert_array_equal(sweep['normalized'],single['normalized'])


def test_failure_manifest_and_no_overwrite(tmp_path, monkeypatch):
    cfg = request(tmp_path, storage={'replay':True})
    def fail(*args, **kwargs):raise RuntimeError('injected storage failure')
    monkeypatch.setattr(run, '_write_traces', fail)
    with pytest.raises(RuntimeError, match='injected'):run.execute(cfg)
    manifest = json.loads((tmp_path/'A/manifest.json').read_text())
    assert manifest['status'] == 'failed'
    before = file_hash(tmp_path/'A/manifest.json')
    with pytest.raises(ValueError, match='already exists'):run.execute(cfg)
    assert file_hash(tmp_path/'A/manifest.json') == before


def test_cwd_paths_validate_does_not_integrate(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):raise AssertionError('validate started simulation')
    monkeypatch.setattr(run, 'simulate', forbidden)
    cfg = request(tmp_path)
    run.inspect_request(cfg)
    path=tmp_path/'nested/study.json';path.parent.mkdir()
    write_json(path, dict(schema='afferent_experiment_run_v1', preset='A_balance_tau50_v1', storage={'output':'../result'}))
    checks=[]
    for cwd in (ROOT,tmp_path):
        result=subprocess.run([sys.executable,str(ROOT/'run.py'),'validate','--config',str(path),'--json'],cwd=cwd,capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        checks.append(json.loads(result.stdout))
    assert checks[0]==checks[1]
    assert checks[0]['effective_config']['storage']['output']==str(tmp_path/'result')


def test_replay_and_compare_are_read_only(tmp_path):
    cfg=request(tmp_path, storage={'replay':True, 'recording_stride':7})
    run.execute(cfg)
    solution=tmp_path/'A/solution.json';before=file_hash(solution)
    replay=resolve_config(solution=solution, action='replay',overrides={'storage':{'output':str(tmp_path/'replay'),'figures':False}})
    run.replay_solution(replay)
    artifacts.compare_runs([tmp_path/'A',tmp_path/'replay'], tmp_path/'comparison')
    assert file_hash(solution)==before
    with np.load(tmp_path/'replay/traces/case_0000.npz') as data:
        assert data['trace'].shape[1]==26
        assert int(data['recording_stride'])==7
        assert str(data['termination']) in ('completed','angle_limit','nonfinite')


def test_d_population_matches_historical_initialization(tmp_path):
    from scripts.AfferentBalance2026.experiments.D_trajectory.run_experiment import Evaluator, train_mode
    c=request(tmp_path, 'C');ctask=run.make_task(c)
    source=tmp_path/'source.json'
    artifacts.save_solution(source,ctask.codec,ctask.codec.encode(ctask.config.initial_physical,np.zeros((2,4))),c)
    d=request(tmp_path,'D',initialization={'solution':str(source)})
    task=run.make_task(d);center,_,_=run._initial_candidates(d,task)
    with Evaluator(task.codec,task.banks,workers=1) as evaluator:
        train_mode(tmp_path/'old',evaluator,center,0.,'joint',d['search'])
    run.execute(d)
    for generation in (1,2):
        with np.load(tmp_path/'old'/f'generation_{generation:03d}.npz') as old, np.load(tmp_path/'D/stages/training/generations'/f'generation_{generation:04d}.npz') as new:
            for a,b in [('normalized','population'),('physical','physical'),('training_scores','scores'),('training_metrics','metrics')]:
                np.testing.assert_array_equal(new[a],old[b])


def test_finetune_generation_zero_selection_and_test_is_after_freeze(tmp_path,monkeypatch):
    cfg=request(tmp_path);task=run.make_task(cfg);source=tmp_path/'source.json'
    artifacts.save_solution(source,task.codec,task.codec.initial(),cfg)
    effective=resolve_config(ROOT/'configs/studies/A_smoke.json',solution=source,action='finetune',
        overrides={'storage':{'output':str(tmp_path/'warm'),'figures':False,'replay':False}})
    calls=[]
    def evaluate(self,population,bank,tau=0.):
        if bank=='test':assert (tmp_path/'warm/selection.json').is_file()
        value=999. if bank=='test' else 1. if not calls else 0.
        calls.append(bank)
        metrics=np.zeros((len(population),len(self.task.banks[bank]),9));metrics[:,:,0]=value
        return np.full(len(population),value),metrics
    monkeypatch.setattr(run.EvaluationPool,'evaluate',evaluate)
    run.execute(effective)
    solution=artifacts.load_solution(tmp_path/'warm/solution.json')
    assert solution.metadata['selection']['generation']==0
    assert solution.metadata['selection']['validation_fitness']==1.
    assert calls[-1]=='test'


def test_prepare_invalid_staged_seed_fails_before_output(tmp_path):
    cfg=request(tmp_path,'C');task=run.make_task(cfg);source=tmp_path/'large_current.json'
    values=np.full((2,4),task.codec.bound)
    artifacts.save_solution(source,task.codec,task.codec.encode(task.config.initial_physical,values),cfg)
    effective=request(tmp_path,'C',search={'method':'staged'},initialization={'solution':str(source)},
                      genome={'command':{'table_current_bound_nA':1.}})
    with pytest.raises(ValueError,match='bounds'):
        run.execute(effective)
    assert not (tmp_path/'C').exists()


def test_schema_types_and_d_replay_configuration(tmp_path):
    for update in ({'search':{'mutation_probability':True}},
                   {'genome':{'overrides':{'neurons.a.0':{'value':'0.1'}}}}):
        with pytest.raises(ValueError,match='finite JSON number'):
            request(tmp_path,**update)
    c=request(tmp_path,'C');task=run.make_task(c);source=tmp_path/'source.json'
    artifacts.save_solution(source,task.codec,task.codec.encode(task.config.initial_physical,np.zeros((2,4))),c)
    d=request(tmp_path,'D',initialization={'solution':str(source)},storage={'replay':True},
              protocol={'replay':{'angles_deg':[-40.,40.],'speeds_rad_s':[-.2,.2], 'duration_s':.01}})
    run.execute(d)
    files=list((tmp_path/'D/traces').glob('case_*.npz'))
    assert len(files)==8  # Two test references, two initial angles and two speeds.
    for path in files:
        with np.load(path) as data:
            spec=json.loads(str(data['case_json']))
            assert spec['initial_angle_deg'] in (-40.,40.)
            assert spec['initial_speed_rad_s'] in (-.2,.2)
            assert spec['duration_s']==.01
            assert len(data['units'])==len(data['columns'])==26
            assert data['reference'].shape[0]==data['trace'].shape[0]
