"""Replay five prespecified distant starts with the saved, validation-selected law.

Run from any directory. This does not train or select a controller. All five
outcomes, including failures if any, are saved for the presentation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

from run_polynomial_study import EXPERIMENTS, REPO, prepare_runtime


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=EXPERIMENTS/'angle_setpoint/report/results/polynomial_five_targets_20261015')
    parser.add_argument('--output',type=Path,default=EXPERIMENTS/'angle_setpoint/report/results/distant_setpoints_20261015')
    args=parser.parse_args();study=args.study.resolve();output=args.output.resolve()
    if (output/'protocol.json').exists():
        raise FileExistsError('Use a new output directory to preserve the existing replay.')
    runtime=prepare_runtime(output)
    sys.path.insert(0,str(REPO));sys.path.insert(0,str(runtime))
    import numpy as np
    from scripts.SteppingReflexGA.experiments.core.polynomial_study import full_command_trace, summarize
    from scripts.SteppingReflexGA.experiments.core.dynamics import TRACE_COLUMNS, METRIC_COLUMNS, rollout
    from scripts.SteppingReflexGA.experiments.core.run import write_json, write_csv, source_hashes

    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    original=json.loads((study/'protocol.json').read_text())
    # Compare against the exact source bytes of training, including the archived library.
    for name,expected in original['source_sha256'].items():
        assert digest(Path(name))==expected,name
    selection=json.loads((study/'selection.json').read_text())
    selected=selection['overall'];law_path=study/selected/'current_law.json'
    law=json.loads(law_path.read_text());coefficients=np.array(law['coefficients'])
    controller=np.load(study/'physical_controller.npy',allow_pickle=False)
    np.testing.assert_array_equal(controller,law['physical_controller'])
    assert law['degree']==3 and original['dt_ms']==.1 and not original['pd_used']
    assert original['hyperparameters']['muscle_torque_factor']==10
    targets=np.array([-45.,-22.5,0.,22.5,45.])
    starts=np.array([45.,45.,-45.,-45.,-45.])
    velocities=np.array([.15,-.15,.15,-.15,.15])
    cases=np.column_stack((np.deg2rad(targets),np.deg2rad(starts),velocities,
                           np.full(5,-1.),np.zeros(5)))
    sources=source_hashes();sources[str(Path(__file__).resolve())]=digest(Path(__file__).resolve())
    sources[str(law_path)]=digest(law_path)
    protocol=dict(study=str(study.relative_to(REPO)),selected_run=selected,
        selection='existing validation winner; no retraining or selection using these cases',
        model=original['model'],dt_ms=.1,duration_s=10.,mode=0,pd_used=False,
        hyperparameters=original['hyperparameters'],source_sha256=sources,
        targets_deg=targets.tolist(),theta0_deg=starts.tolist(),omega0_rad_s=velocities.tolist(),
        initial_error_deg=(starts-targets).tolist(),kick_used=False,cases=5)
    write_json(output/'protocol.json',protocol)
    write_csv(output/'cases.csv',[dict(trial=i,target_deg=targets[i],theta0_deg=starts[i],
        omega0_rad_s=velocities[i],kick_time_s=-1.,kick_delta_omega=0.) for i in range(5)])
    metrics=[];records=[];(output/'traces').mkdir(exist_ok=True)
    for i,case in enumerate(cases):
        m,trace,commanded=full_command_trace(controller,coefficients,case)
        scalar,_=rollout(commanded,case[1:],target_rad=case[0],duration_s=10.,stride=0,mode=0)
        np.testing.assert_allclose(m,scalar,rtol=1e-12,atol=1e-12)
        path=output/f'traces/case_{i:03d}.npz'
        np.savez_compressed(path,trace=trace,columns=np.array(TRACE_COLUMNS),case=case,
            physical_controller=commanded,fixed_controller=controller,coefficients=coefficients,
            metrics=m,metric_columns=np.array(METRIC_COLUMNS),model=original['model'],dt_ms=.1,mode=0)
        metrics.append(m)
        record=dict(trial=i,target_deg=targets[i],theta0_deg=starts[i],omega0_rad_s=velocities[i],
            initial_error_deg=starts[i]-targets[i],samples=len(trace),duration_s=float(trace[-1,0]),
            success=bool(m[5]),settling_s=float(m[6]),late_rmse_deg=float(np.rad2deg(m[7])),
            file=str(path.relative_to(output)),sha256=digest(path))
        records.append(record);print(json.dumps(record),flush=True)
    metrics=np.array(metrics)
    write_csv(output/'metrics.csv',[dict(trial=i,**dict(zip(METRIC_COLUMNS,row.tolist()))) for i,row in enumerate(metrics)])
    summary=dict(protocol=protocol,metrics=summarize(metrics),traces=records)
    write_json(output/'summary.json',summary)
    for name,expected in sources.items():assert digest(Path(name))==expected,name
    with zipfile.ZipFile(output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            path=Path(name);archive.write(path,str(path.relative_to(REPO)))
    write_json(output/'verification.json',dict(status='passed',cases=5,
        full_trace_and_metrics_only_agree=True,trace_recomputed_metrics_agree=True,
        fixed_controller_and_current_law_unchanged=True,source_hashes_verified=len(sources)))
    print(json.dumps(summary['metrics'],indent=2),flush=True)


if __name__=='__main__':main()
