"""Study bounds and activity diagnostics, independent of archived controllers."""
from pathlib import Path
import sys

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1]/'topubsrc'
sys.path.insert(0, str(SOURCE))
from evolution import profile_inputs, make_genome, initial_population, reproduce, seed_coordinates
from model import measure_activity, TRACE_COLUMNS


@pytest.mark.parametrize('folder,profile', [
    ('A_balance','fast'),('A_balance','slow'),('B_balance','fast'),('B_balance','slow'),
    ('C_setpoint_balance','fast'),('C_setpoint_balance','slow'),('C_setpoint_balance','mirror')])
def test_strict_cross_bounds_survive_initialization_and_reproduction(folder,profile):
    config,_=profile_inputs(SOURCE/folder,profile)
    kinds=['table','polynomial'] if folder.startswith('C') else [None]
    for kind in kinds:
        genome=make_genome(config,kind)
        rng=np.random.default_rng(42)
        population=initial_population(seed_coordinates(genome),50,rng)
        for _ in range(3):
            physical=np.array([genome.decode(row)[0] for row in population])
            assert np.all(physical[:,[49,50]]>=0) and np.all(physical[:,[49,50]]<.1)
            population=reproduce(population,np.linspace(0,1,50),rng,.2,.3)
        endpoint=genome.decode(np.ones(genome.size))[0]
        np.testing.assert_array_equal(endpoint[[49,50]],np.nextafter(.1,0.))
        np.testing.assert_array_equal(endpoint[[48,51]],[2.,2.])
        invalid=endpoint.copy();invalid[49]=.1
        with pytest.raises(ValueError):genome.controller.encode(invalid)


def trace_fixture(end=10.):
    trace=np.zeros((round(end*10000)+1,len(TRACE_COLUMNS)))
    trace[:,0]=np.arange(len(trace))/10000.
    return dict(trace=trace,metrics=np.array([1,end,0,0,0,0,0,0,0]),duration_s=10.,fitness_window_s=2.)


def test_common_bins_do_not_require_coincident_spikes_and_detect_silence():
    result=trace_fixture();trace=result['trace']
    trace[[81000,91000],19]=30
    trace[[82000,92000],20]=30
    report=measure_activity(result)
    assert report['both_active_every_late_bin']
    assert report['late_bin_spike_counts']==[[1,1],[1,1]]
    assert report['late_rate_hz']==[1.,1.]
    trace[:,20]=0
    report=measure_activity(result)
    assert not report['both_active_late'] and not report['both_active_every_late_bin']
    assert report['late_rate_hz']==[1.,0.]


def test_bin_boundary_is_counted_once_and_unfinished_window_is_not_zero_activity():
    result=trace_fixture()
    result['trace'][90000,19:21]=30
    report=measure_activity(result)
    assert report['late_bin_spike_counts']==[[1,1],[0,0]]
    assert report['both_active_late'] and not report['both_active_every_late_bin']
    assert report['coactive_bin_fraction']==.5
    truncated=measure_activity(trace_fixture(8.5))
    assert not truncated['completed'] and truncated['late_rate_hz'] is None


def test_decimated_trace_cannot_measure_spike_activity():
    result=trace_fixture();result['trace']=result['trace'][::20]
    with pytest.raises(ValueError,match='stride=1'):measure_activity(result)


def test_activity_replay_preserves_exact_si_bank_including_perturbation(tmp_path,monkeypatch):
    """Diagnostics must not perturb a chaotic replay via angle unit conversion."""
    import model
    import run
    config,_=profile_inputs(SOURCE/'A_balance','fast')
    genome=make_genome(config)
    genome.save(tmp_path/'best_genome.json',seed_coordinates(genome))
    stage=tmp_path/'training';stage.mkdir()
    rng=np.random.default_rng(123)
    theta=next(x for x in rng.uniform(-.7,.7,100) if np.deg2rad(np.rad2deg(x))!=x)
    case=np.array([0.,theta,.123456789,.2,.1])
    np.savez(stage/'cases.npz',test=case[None])
    replay=trace_fixture();np.savez(tmp_path/'test_metrics.npz',metrics=replay['metrics'][None])
    seen=[]
    def simulate(physical,actual_case,mechanics,**kwargs):
        seen.append(actual_case.copy())
        return replay['metrics'],replay['trace']
    monkeypatch.setattr(model,'simulate',simulate)
    result=run.activity(tmp_path)
    np.testing.assert_array_equal(seen[0],case)
    assert result['complete_late_windows']==1
    assert result['silent_late_cases']==[1,1]
    assert result['used_for_selection'] is False
