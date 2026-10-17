"""Run the five-target quadratic/cubic study with an isolated, archived library."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

REPO=Path(__file__).resolve().parents[5]
EXPERIMENTS=REPO/'scripts/SteppingReflexGA/experiments'
DEFAULT_POPULATION=EXPERIMENTS/'small_nm_weights/report/results/sweep_population_mutation_20261008/mutation_0.300/population_050/seed_42/epochs_200/final_population.npz'
REFERENCE_LIBRARY_SHA256='2b2ff19c7fbd6fc0bd7ddbb74a9ed46cee45b2592cfd22d4332da621f35c4376'


def prepare_runtime(output):
    output.mkdir(exist_ok=True,parents=True)
    runtime=output/'runtime'
    archive=EXPERIMENTS/'small_nm_weights/report/results/wide_replay_20261009/source_snapshot.zip'
    if archive.exists():
        run=json.loads(archive.with_name('run.json').read_text())
        with zipfile.ZipFile(archive) as z:
            name=next(n for n in z.namelist() if n.endswith('src/SpikingNN/core/Izh_net.py'))
            content=z.read(name)
        expected=next(v for k,v in run['source_sha256'].items() if k.endswith('src/SpikingNN/core/Izh_net.py'))
        source_info=dict(original_source_archive=str(archive.relative_to(REPO)))
    else:
        # A code-only checkout has no saved runs. Accept the library only if it
        # exactly matches the verified model; never silently change mechanics.
        library=REPO/'src/SpikingNN/core/Izh_net.py'
        content=library.read_bytes();expected=REFERENCE_LIBRARY_SHA256
        if hashlib.sha256(content).hexdigest()!=expected:
            raise RuntimeError('No saved runtime: the working Izh_net.py differs from the verified model. '
                               'Restore the matching library or provide the historical source archive.')
        source_info=dict(original_source_library=str(library.relative_to(REPO)))
    assert hashlib.sha256(content).hexdigest()==expected
    if not runtime.exists():
        shutil.copytree(REPO/'src/SpikingNN',runtime/'SpikingNN',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        (runtime/'SpikingNN/core/Izh_net.py').write_bytes(content)
        (output/'runtime_provenance.json').write_text(json.dumps(dict(
            **source_info,Izh_net_sha256=expected,
            reason='Use the verified full-Afferents mechanics of the trained controller; preserve working library edits.'),indent=2)+'\n')
    assert hashlib.sha256((runtime/'SpikingNN/core/Izh_net.py').read_bytes()).hexdigest()==expected
    return runtime


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--population-from',type=Path,default=DEFAULT_POPULATION)
    parser.add_argument('--population',type=int,default=50)
    parser.add_argument('--generations',type=int,default=150)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43])
    parser.add_argument('--workers',type=int,default=10)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    runtime=prepare_runtime(args.output.resolve())
    sys.path.insert(0,str(REPO));sys.path.insert(0,str(runtime))
    from scripts.SteppingReflexGA.experiments.core.polynomial_study import run_study
    run_study(args.output,args.population_from,population=args.population,generations=args.generations,
              seeds=args.seeds,workers=args.workers,resume=args.resume)


if __name__=='__main__':main()
