"""Scientific checks: API parity, fixed-phase fitness, serialization and GA resume.

Run: python -m unittest -v test_training
"""
from copy import deepcopy
from pathlib import Path
import unittest
import numpy as np
import numba

from simulate import build_config
from dynamics import pack, replay
from fitness import reference, task_vector, measure, evaluate_population, diagnostics
from evolution import optimize
from train import read_json, SearchSpace, verify_api

HERE=Path(__file__).resolve().parent


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system=read_json(HERE/'system.json')
        cls.genome=read_json(HERE/'genome.json')
        cls.settings=read_json(HERE/'training.json')
        cls.space=SearchSpace(cls.system,cls.genome,cls.settings['search'])
        numba.set_num_threads(min(4,numba.config.NUMBA_NUM_THREADS))

    def test_api_state_parity_and_tonic_drive(self):
        base=build_config(self.system,self.genome)
        for drive in (.5,1.,1.5):
            errors,_=verify_api(base,duration_s=.5,drive=drive)
            self.assertLess(errors['q'],1e-8)
        modified=deepcopy(base)
        modified['network']['tau_syn'][1][2]*=1.3
        modified['network']['weights'][1][2]*=.8
        modified['network']['Q_aff'][1][3]*=1.7
        modified['limbs'][0]['extensor']['A']*=1.2
        modified['limbs'][0]['flexor']['tau_c']*=.7
        modified['limbs'][0]['mechanics'].update(q0=1.4,w0=.2)
        verify_api(modified,duration_s=.5)

    def test_batch_matches_replay_and_threads(self):
        rng=np.random.default_rng(12)
        units=np.vstack((self.space.center,np.clip(self.space.center+rng.normal(0,.015,(3,22)),0,1)))
        physical=self.space.decode(units);spec=task_vector(self.settings['task'])
        numba.set_num_threads(1)
        serial=evaluate_population(physical,spec)
        numba.set_num_threads(min(4,numba.config.NUMBA_NUM_THREADS))
        parallel=evaluate_population(physical,spec)
        np.testing.assert_array_equal(serial,parallel)
        for p,expected in zip(physical,serial):
            for detailed in (False,True):
                tr,n,f=replay(p,duration_s=10,stride=10,detailed=detailed,angle_limit_deg=85)
                np.testing.assert_array_equal(measure(tr,n,f,spec),expected)

    def test_physical_genome_roundtrip(self):
        rng=np.random.default_rng(3)
        for unit in (self.space.center,np.zeros(22),np.ones(22),rng.random(22)):
            document=self.space.payload(unit,{})
            restored=pack(build_config(self.system,document))
            np.testing.assert_array_equal(restored,self.space.decode(unit)[0])
        self.assertEqual(self.genome,read_json(HERE/'genome.json'))

    def test_fixed_phase_objective(self):
        task=self.settings['task'];spec=task_vector(task)
        t=np.arange(10000)*.001
        q,w=reference(t,task)
        trace=np.column_stack((t,q,w,np.zeros((len(t),2))))
        ideal=measure(trace,100000,0,spec)
        self.assertAlmostEqual(ideal[0],1.)
        self.assertAlmostEqual(ideal[1],0.)
        shifted=deepcopy(task);shifted['phase_rad']=np.pi
        q,w=reference(t,shifted)
        trace[:,1],trace[:,2]=q,w
        wrong=measure(trace,100000,0,spec)
        self.assertAlmostEqual(wrong[1],np.sqrt(2)*task['amplitude_deg'],places=9)
        self.assertLess(wrong[0],.4)
        self.assertEqual(measure(trace,50000,1,spec)[0],-1)
        d=diagnostics(trace,task)
        self.assertAlmostEqual(abs(d['phase_error_deg']),180.,places=8)
        self.assertAlmostEqual(d['fitted_target_amplitude_deg'],10.,places=8)
        self.assertLess(abs(d['dominant_motion_frequency_hz']-1),.01)

    def test_invalid_configuration(self):
        for key,value in [('frequency_hz',0),('amplitude_deg',0),('phase_rad',float('nan')),
                          ('score_start_s',10),('duration_s',2.5),('tonic_drive',-1)]:
            task=deepcopy(self.settings['task']);task[key]=value
            with self.assertRaises(ValueError):task_vector(task)
        bad=deepcopy(self.settings['search']);bad['groups'][1]['targets']=bad['groups'][0]['targets']
        with self.assertRaises(ValueError):SearchSpace(self.system,self.genome,bad)

    def test_ga_checkpoint_resume(self):
        settings=dict(sigma_start=.12,sigma_end=.004,mutation_probability=.25)
        def evaluate(population):
            return (1./(1.+((population-.3)**2).sum(axis=1)))[:,None]
        arguments=dict(population=12,generations=12,seed=18,settings=settings)
        complete=optimize(np.array([.5,.5]),evaluate,**arguments)
        saved={}
        class Stop(Exception):pass
        def checkpoint(state):
            if state['generation']==5:
                saved.update(deepcopy(state));raise Stop
        with self.assertRaises(Stop):optimize(np.array([.5,.5]),evaluate,callback=checkpoint,**arguments)
        resumed=optimize(np.array([.5,.5]),evaluate,resume=saved,**arguments)
        np.testing.assert_array_equal(resumed['population'],complete['population'])
        np.testing.assert_array_equal(resumed['scores'],complete['scores'])


if __name__=='__main__':unittest.main()
