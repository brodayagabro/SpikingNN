"""Build the bilingual scientific article from frozen retraining artifacts."""
import argparse
import csv
import json
from pathlib import Path
import shlex
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
from scripts.AfferentBalance2026.experiments.visualization import configure_fonts
from scripts.AfferentBalance2026.tools.recheck_tensile_genomes import load_candidate,digest,write_json
from scripts.AfferentBalance2026.experiments.core.fitness import fitness

HERE=Path(__file__).resolve().parent
EXPERIMENTS=HERE.parent/'experiments'
STUDY = 'muscle_tau50_20261026'
OUTPUT = HERE/'results'/STUDY
MIRROR_STUDY = 'mirror_symmetry_20261026'



def read_json(path):
    """Read a UTF-8 calculation artifact without modifying it."""
    return json.loads(Path(path).read_text())


def relative(path):
    """Return a Markdown path relative to the report directory."""
    import os
    return Path(os.path.relpath(path,HERE)).as_posix()


def figure(path,caption):
    """Create an explicit Markdown image and caption, also supported by Pandoc."""
    return f'![{caption}]({relative(path)})\n\n{caption}\n'


def save_plot(fig,name):
    """Save publication figures in both raster and vector formats."""
    for suffix in ('png','pdf'):fig.savefig(OUTPUT/f'{name}.{suffix}',dpi=180)
    plt.close(fig)
    return OUTPUT/f'{name}.png'


def mechanics():
    """Draw actual tensile-force directions and a compact closed-loop diagram."""
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    ax=axes[0];theta=np.deg2rad(25);rod=.3*np.array([np.sin(theta),np.cos(theta)]);point=rod/6
    ax.plot([0,rod[0]],[0,rod[1]],color='black',lw=5)
    for anchor,color,label in ((np.array([.2,0]),'crimson','Flexor'),(np.array([-.2,0]),'royalblue','Extensor')):
        ax.plot([anchor[0],point[0]],[anchor[1],point[1]],color=color,lw=2)
        direction=(anchor-point)/np.linalg.norm(anchor-point)
        end=point+.10*direction
        ax.annotate('',end,point,arrowprops=dict(arrowstyle='->',lw=3,color=color))
        ax.plot(*anchor,'s',color=color);ax.text(anchor[0],-.025,label,ha='center',color=color)
    center=rod/2
    ax.annotate('',center+[0,-.09],center,arrowprops=dict(arrowstyle='->',lw=3,color='darkgreen'))
    ax.text(center[0]+.015,center[1]-.06,'mg',color='darkgreen')
    ax.plot([0,0],[0,.31],'k--',alpha=.4);ax.plot(0,0,'ko');ax.text(.012,.23,r'$\theta$')
    ax.set(xlim=(-.26,.26),ylim=(-.05,.33),aspect='equal',title='Tension points towards each anchor')
    ax.axis('off');ax.text(-.25,.31,'A',weight='bold',fontsize=20)
    ax=axes[1];ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    boxes=[(.05,.6,'Desired angle'),(.4,.6,'Polynomial\ncurrents'),(.75,.6,'2 spiking\nneurons'),(.75,.18,'2 muscles'),(.4,.18,'Pendulum'),(.05,.18,'6 afferent\nchannels')]
    for x,y,text in boxes:ax.text(x+.10,y+.09,text,ha='center',va='center',fontsize=13,bbox=dict(boxstyle='round',facecolor='#edf2f7',pad=.6))
    for start,end in (((.27,.69),(.37,.69)),((.62,.69),(.72,.69)),((.85,.56),(.85,.37)),((.72,.27),(.63,.27)),((.37,.27),(.28,.27)),((.15,.39),(.15,.48)),((.15,.48),(.85,.48)),((.85,.48),(.85,.56))):
        ax.annotate('',end,start,arrowprops=dict(arrowstyle='->',color='black'))
    ax.text(.05,.92,'B',weight='bold',fontsize=20);ax.text(.48,.04,'A/B: applied currents are zero',ha='center',fontsize=12)
    return save_plot(fig,'mechanics')


def analytics():
    """Evaluate independently derived mechanical control and force-capacity estimates."""
    m,l,b,g=.3,.3,.001,9.81;J=m*l*l/3;kg=m*g*l/2;a1,a2=.2,.05
    roots=np.roots([J,b,-kg]);angles=np.array([-45.,-22.5,0.,22.5,45.]);theta=np.deg2rad(angles)
    length_f=np.sqrt(a1*a1+a2*a2-2*a1*a2*np.sin(theta));length_e=np.sqrt(a1*a1+a2*a2+2*a1*a2*np.sin(theta))
    rf=a1*a2*np.cos(theta)/length_f;re=a1*a2*np.cos(theta)/length_e
    torque=-kg*np.sin(theta);force=np.where(torque>=0,torque/rf,-torque/re)
    result=dict(inertia=J,gravity_coefficient=kg,open_loop_eigenvalues=roots.tolist(),
                controllability_determinant=-1/J**2,angles_deg=angles.tolist(),required_torque_Nm=torque.tolist(),
                minimum_single_muscle_force_N=force.tolist())
    write_json(OUTPUT/'analytical_estimates.json',result);return result


def load_runs(allow_incomplete):
    """Require completed runs and cross-check scalar fitness against saved episode matrices."""
    rows=[]
    paths = sorted(EXPERIMENTS.glob(f'*/results/{STUDY}/*/protocol.json'))
    if not paths:
        raise FileNotFoundError(f'No training protocols found for {STUDY}')
    for protocol in paths:
        path = protocol.parent
        experiment, name = path.parents[2].name, path.name
        label = f'{experiment[0]} {name}'
        if not (path/'result.json').exists():
            if allow_incomplete:continue
            raise FileNotFoundError(path/'result.json')
        result=read_json(path/'result.json');assert result['status']=='completed'
        genome = read_json(path/'best_genome.json')
        times = np.asarray(genome['physical'])[[19, 20, 26, 27]]
        assert np.all(times >= 50.)
        assert all(genome['configuration']['parameters'][i]['minimum'] == 50. for i in (19,20,26,27))
        options=read_json(path/'protocol.json')['options'];test=np.load(path/'test_metrics.npz')
        metrics=test['metrics'];np.testing.assert_allclose(fitness(metrics),test['scores'][0],atol=1e-12)
        stages=[path/'table',path/'polynomial'] if options['method']=='staged' else [path/'training']
        for stage in stages:
            run=read_json(stage/'run.json');assert run['status']=='completed'
            for snap in (stage/'generations').glob('generation_*.npz'):
                with np.load(snap) as data:
                    assert np.isfinite(data['physical']).all()
                    assert np.all(data['physical'][:, [19,20,26,27]] >= 50.)
                    assert np.all(data['physical'][:, [19,26]] <= 100.)
                    assert np.all(data['physical'][:, [20,27]] <= 200.)
                    assert data['physical'].shape[0]==options['population']
            assert len(list((stage/'generations').glob('generation_*.npz')))==run['generations']
        durations=np.full(len(metrics),10.)
        if experiment=='D_trajectory':
            banks=read_json(stages[0]/'banks.json');durations=np.array([s['duration_s'] for s in banks['test']])
        survived=np.isclose(metrics[:,1],durations,atol=1e-9,rtol=0)&(metrics[:,0]>=0)
        if experiment!='D_trajectory':survived &= abs(metrics[:,8])<=np.pi/2
        rows.append(dict(label=label,experiment=experiment,name=name,path=path,result=result,options=options,stages=stages,
                         metrics=metrics,survived=int(survived.sum()),success=int(metrics[:,5].sum()),count=len(metrics),test_fitness=fitness(metrics)))
    return rows


def training_plot(rows):
    """Plot the complete recorded training and validation histories for every branch."""
    height = max(2, int(np.ceil(len(rows)/3)))
    fig,axes=plt.subplots(height,3,figsize=(13,3.5*height),constrained_layout=True)
    for ax,row in zip(axes.flat,rows):
        offset=0
        for stage in row['stages']:
            with (stage/'history.csv').open() as f:history=list(csv.DictReader(f))
            with (stage/'validation_history.csv').open() as f:val=list(csv.DictReader(f))
            ax.plot([int(x['generation'])+offset for x in history],[float(x['best_fitness']) for x in history],lw=1,label='Training' if offset==0 else None)
            ax.plot([int(x['generation'])+offset for x in val],[float(x['selected_fitness']) for x in val],lw=1.5,label='Validation' if offset==0 else None)
            if offset:ax.axvline(offset,color='grey',ls=':')
            offset+=len(history)
        ax.set(title=row['label'],xlabel='Generation',ylabel='Fitness',ylim=(-.02,1.02));ax.grid(alpha=.2);ax.legend(fontsize=10)
    for ax in list(axes.flat)[len(rows):]:ax.axis('off')
    return save_plot(fig,'training_histories')


def tables(rows,en):
    """Report every branch, including unsuccessful starts and extra optimization budgets."""
    lines=['| Run | N × G | Validation before → after | Test fitness | Test completion | Test success |',
           '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        r,o=row['result'],row['options']
        lines.append(f'| {row["label"]} | {o["population"]} × {o["generations"]} | {r["baseline_validation_fitness"]:.4f} → {r["validation_fitness"]:.4f} | {row["test_fitness"]:.4f} | {row["survived"]}/{row["count"]} | {row["success"]}/{row["count"]} |')
    note=('For staged C, the displayed baseline is the polynomial-stage start after least squares; the total budget includes both stages. Warm starts and refinements have different histories and are not an equal-budget comparison. All four muscle times are at least 50 ms. Completion and strict success are reported separately.' if en else
          'Для двухэтапного C исходная оценка относится к началу полиномиальной стадии после МНК; полный бюджет включает обе стадии. Стартовые решения и уточнения имеют разную историю и не образуют сравнение при равном суммарном бюджете. Все четыре мышечных времени не менее 50 мс. Полное прохождение и строгий успех показаны отдельно.')
    return '\n'.join(lines)+'\n\n'+note



def detailed_results(rows, selected, en):
    """Explain completion, accuracy and velocity criteria using the selected saved matrices."""
    generations=sum(r['options']['generations'] for r in rows)
    evaluations=sum(r['options']['generations']*r['options']['population'] for r in rows)
    lines=[(f'The series comprises {len(rows)} branches, {generations} evaluated generations and {evaluations} training-candidate evaluations; validation and test evaluations are additional.' if en else
            f'Серия содержит {len(rows)} веток, {generations} оценённых поколений и {evaluations} оценок учебных особей; проверки и тесты выполняются дополнительно.')]
    lines+=['','| Task | Grid success | Near-target success | Mean late RMSE of completed tests (deg) |', '|---|---:|---:|---:|']
    for key,row in selected.items():
        path=row['path']/'figures/summary.json'
        if not path.exists() or key=='D':continue
        summary=read_json(path);grid=summary['grid'];near=summary['near_target']
        metrics=row['metrics'];mask=np.isclose(metrics[:,1],10.,rtol=0,atol=1e-9)&(abs(metrics[:,8])<=np.pi/2)
        late=float(np.rad2deg(metrics[mask,7]).mean()) if mask.any() else float('nan')
        lines.append(f'| {key} | {sum(x["successes"] for x in grid)}/{sum(x["cases"] for x in grid)} | {near["successes"]}/{near["cases"]} | {late:.4f} |')
    if 'C' in selected:
        row=selected['C'];metrics=row['metrics'];failed=np.flatnonzero(metrics[:,1]<10.-1e-9)
        bank=np.load(row['stages'][-1]/'cases.npz')['test']
        if len(failed) > 5:
            lines += ['', f'C: {len(failed)} incomplete tests; first five cases are listed below, all remain in test_metrics.npz.']
        for index in failed[:5]:
            target,initial,speed=bank[index,:3]
            lines+=['',(f'C failure: target {np.rad2deg(target):.2f}°, initial angle {np.rad2deg(initial):.2f}°, initial velocity {speed:.4f} rad/s; termination at {metrics[index,1]:.4f} s. This case remains included in every aggregate.' if en else
                         f'Неудачный тест C: цель {np.rad2deg(target):.2f}°, начальный угол {np.rad2deg(initial):.2f}°, скорость {speed:.4f} рад/с; остановка на {metrics[index,1]:.4f} с. Этот случай включён во все общие оценки.')]
    if 'C' in selected:
        lines += ['', ('C diagnostics for the five plotted examples (initial −30°, zero speed), after 3 s:' if en else
                       'Диагностика C для пяти показанных примеров (начальный угол −30°, нулевая скорость) после 3 с:'), '',
                  '| Target (deg) | Maximum absolute error (deg) | Maximum speed magnitude (rad/s) |',
                  '|---:|---:|---:|']
        for path in sorted((selected['C']['path']/'figures').glob('*_example.npz')):
            with np.load(path) as stored:
                trace, case = stored['trace'], stored['case']
            late = trace[trace[:,0] >= 3.]
            if len(late):
                lines.append(f'| {np.rad2deg(case[0]):g} | {np.rad2deg(abs(late[:,1]-case[0])).max():.3f} | {abs(late[:,2]).max():.3f} |')
        lines += ['', ('The strict limits are 3° and 0.35 rad/s throughout the interval. Full completion and low mean angle error do not imply meeting these instantaneous limits.' if en else
                       'Строгие пределы составляют 3° и 0,35 рад/с на всём интервале. Полное прохождение и небольшая средняя ошибка угла не означают соблюдения этих мгновенных ограничений.')]
    if 'D' in selected:
        m=selected['D']['metrics'];rmse=np.rad2deg(m[:,2]);fraction=m[:,6]
        lines += ['', (f'D mean angle RMSE: {rmse.mean():.3f}° (range {rmse.min():.3f}–{rmse.max():.3f}°); mean velocity-error RMSE: {m[:,7].mean():.3f} rad/s. Joint angle/velocity band occupancy: {100*fraction.min():.1f}–{100*fraction.max():.1f}%; strict success requires at least 95%, completion and angle RMSE ≤3°.' if en else
                       f'В D средний RMSE угла: {rmse.mean():.3f}° (диапазон {rmse.min():.3f}–{rmse.max():.3f}°); средний RMSE ошибки скорости: {m[:,7].mean():.3f} рад/с. Доля времени одновременного соблюдения ограничений угла и скорости: {100*fraction.min():.1f}–{100*fraction.max():.1f}%; строгий успех требует не менее 95%, полного прохождения и RMSE угла ≤3°. Для оборванных траекторий ошибка относится к пройденной части оценочного интервала; если этот интервал ещё не начался, используются штрафы 90° и 10 рад/с. Ошибка не заменяет показатель прохождения.')]
    if 'D' in selected:
        recorded=selected['D']['path']/'figures/trajectory_tests.npz'
        if recorded.exists():
            with np.load(recorded) as data:lowest=float(data['trace_000'][:,5:7].min())
            lines += ['', (f'The first D example reaches a minimum membrane potential of {lowest:.1f} mV. The current fitness does not constrain membrane range, firing frequency or energy; numerical tracking performance alone is not physiological validation.' if en else
                           f'В первом примере D минимальный мембранный потенциал равен {lowest:.1f} мВ. Текущая приспособленность не ограничивает диапазон потенциалов, частоту импульсов или энергозатраты; численное качество слежения само по себе не является физиологической валидацией.')]
    confirmation = OUTPUT/'D_confirmation.json'
    if confirmation.exists():
        x = read_json(confirmation)['summary']
        lines += ['', (f'Fresh D confirmation bank (seed {x["seed"]}): completed {x["completed"]}/{x["cases"]}; strict success {x["success"]}/{x["cases"]}; aggregate fitness {x["fitness"]:.4f}. No additional training or selection follows this evaluation.' if en else
                       f'Новый подтверждающий банк D (seed {x["seed"]}): завершено {x["completed"]}/{x["cases"]}; строгий успех {x["success"]}/{x["cases"]}; приспособленность {x["fitness"]:.4f}. После этой проверки обучение и выбор генома не продолжаются.')]
    return '\n'.join(lines)


def timescale_comparison(selected, en):
    """Compare saved old winners with the new series on unchanged held-out banks."""
    old = read_json(HERE/'results/tensile_retrain_20261026/manifest.json')['selected']
    lines = [('Comparison with the previous tensile-mechanics winners; banks and success criteria are unchanged.' if en else
              'Сравнение с прежними победителями на механике натяжения: банки и критерии успеха сохранены.'), '',
             '| Task | Old τc F/E (ms) | New τc F/E (ms) | Old τ1 F/E (ms) | New τ1 F/E (ms) | Old → new strict test success |',
             '|---|---:|---:|---:|---:|---:|']
    for key, row in selected.items():
        previous = REPO/old[key]['run']
        old_stage = previous/('polynomial' if (previous/'polynomial').exists() else 'training')
        new_stage = row['stages'][-1]
        if key == 'D':
            assert read_json(old_stage/'banks.json')['test'] == read_json(new_stage/'banks.json')['test']
        else:
            with np.load(old_stage/'cases.npz') as a, np.load(new_stage/'cases.npz') as b:
                np.testing.assert_array_equal(a['test'], b['test'])
        p = np.asarray(read_json(previous/'best_genome.json')['physical'])
        q = np.asarray(read_json(row['path']/'best_genome.json')['physical'])
        with np.load(previous/'test_metrics.npz') as stored:
            old_metrics = stored['metrics']
        pairs = [f'{x[a]:.3f}/{x[b]:.3f}' for x,a,b in ((p,19,26),(q,19,26),(p,20,27),(q,20,27))]
        lines.append(f'| {key} | '+ ' | '.join(pairs)+f' | {int(old_metrics[:,5].sum())}/{len(old_metrics)} → {row["success"]}/{row["count"]} |')
    lines += ['', ('Force variation in the same example: target 0°, initial angle −30°, zero initial velocity; statistics over seconds 8–10. These are single-example diagnostics, not population statistics.' if en else
                   'Вариация силы в одинаковом примере: цель 0°, начальный угол −30°, начальная скорость нулевая; статистика за 8–10 с. Это диагностика одного примера, а не статистика популяции.'), '',
              '| Task / version | Mean F flex/ext (N) | SD F flex/ext (N) | Angle SD (deg) |',
              '|---|---:|---:|---:|']
    force_rows = []
    for key, row in selected.items():
        if key == 'D':
            continue
        for label, run in (('old', REPO/old[key]['run']), ('new', row['path'])):
            path = run/'figures/target_+000.0_example.npz'
            if not path.exists():
                continue
            with np.load(path) as stored:
                trace = stored['trace']
            complete = bool(np.isclose(trace[-1,0], 10., atol=1e-9))
            late = trace[trace[:,0] >= 8.]
            record = dict(experiment=key, version=label, complete=complete, source=str(path.relative_to(REPO)))
            if complete:
                mean = late[:,3:5].mean(axis=0); sd = late[:,3:5].std(axis=0)
                angle_sd = float(np.rad2deg(late[:,1]).std())
                record.update(force_mean_N=mean.tolist(), force_sd_N=sd.tolist(), angle_sd_deg=angle_sd)
                lines.append(f'| {key} / {label} | {mean[0]:.3f}/{mean[1]:.3f} | {sd[0]:.3f}/{sd[1]:.3f} | {angle_sd:.4f} |')
            else:
                lines.append(f'| {key} / {label} | incomplete | incomplete | incomplete |')
            force_rows.append(record)
    write_json(OUTPUT/'force_variation_comparison.json', force_rows)
    lines += ['', ('Other genes were also retrained, so this comparison does not isolate the causal effect of changing time constants.' if en else
                   'Другие гены также переобучались, поэтому сравнение не выделяет изолированный причинный эффект изменения временных констант.')]
    return '\n'.join(lines)


def verification_text(en):
    """Read completed checks rather than carrying verification claims across series."""
    path = OUTPUT/'verification.json'
    if not path.exists():
        return 'Verification is pending; this is a draft report.' if en else 'Проверки ещё не завершены; это черновик отчёта.'
    data = read_json(path)
    return data['summary_en' if en else 'summary_ru']


def selected_runs(rows):
    """Select a winner per task by validation within the common 50 ms muscle-time domain."""
    selected={}
    for key in 'ABCD':
        candidates=[r for r in rows if r['experiment'].startswith(key+'_') ]
        if candidates:selected[key]=max(candidates,key=lambda r:r['result']['validation_fitness'])
    return selected


def mirror_materials():
    """Verify published mirror results against frozen genomes and episode arrays.

    This reads completed calculations only. It neither trains controllers nor
    republishes canonical genomes, and keeps the asymmetric series separate.
    """
    from scripts.AfferentBalance2026.tools.summarize_mirror_study import metrics_summary

    source = HERE/'results'/MIRROR_STUDY/'comparison.json'
    results = read_json(source)
    provenance = {'comparison': relative(source), 'comparison_sha256': digest(source),
                  'selected': {}, 'files': {}}
    for key, experiment in (('C', 'C_setpoint_balance'), ('D', 'D_trajectory')):
        record = results[key]
        root = EXPERIMENTS/experiment/'results'/MIRROR_STUDY
        trained = [row for row in record['rows'] if row['branch'] != 'asymmetric_baseline']
        assert record['selected'] == max(trained, key=lambda row: row['validation_fitness'])['branch']
        for row in record['rows']:
            run = root/row['branch']
            candidate = load_candidate(experiment, row['branch'], run/'best_genome.json')
            assert digest(run/'best_genome.json') == row['genome_sha256']
            with np.load(run/'test_metrics.npz') as stored:
                actual = metrics_summary(stored['metrics'], candidate['cfg'], key)
            for name, value in actual.items():
                np.testing.assert_allclose(value, row[name], rtol=0, atol=1e-10)
            provenance['files'][relative(run/'test_metrics.npz')] = digest(run/'test_metrics.npz')
            if row['branch'] != 'asymmetric_baseline':
                assert candidate['cfg'].mirror_symmetry
                protocol = read_json(run/'protocol.json')
                assert read_json(run/'result.json')['status'] == 'completed'
                row['protocol'] = protocol
        winner = root/record['selected']
        figure_summary = read_json(winner/'figures/summary.json')
        assert figure_summary['genome_sha256'] == digest(winner/'best_genome.json')
        assert record['confirmation'] == read_json(root/'confirmation/summary.json')
        assert record['confirmation']['genome_sha256']['symmetric'] == digest(winner/'best_genome.json')
        assert not record['confirmation']['selection_used']
        provenance['selected'][key] = dict(run=relative(winner),
                                           genome_sha256=digest(winner/'best_genome.json'))
        figures = (['target_+045.0.png', 'target_+045.0.pdf', 'target_+045.0_grid.npz',
                    'target_+045.0_zero_speed.npz', 'target_+045.0_example.npz']
                   if key == 'C' else ['tracking.png', 'tracking.pdf', 'trajectory_tests.npz'])
        for path in [root/'confirmation/summary.json', winner/'figures/summary.json',
                     *(winner/'figures'/name for name in figures)]:
            provenance['files'][relative(path)] = digest(path)
    return results, provenance


def mirror_tables(results, en):
    """Render all mirror branches and fresh-bank outcomes without dropping failures."""
    lines = [('The same held-out banks as the asymmetric ≥50 ms series are used below.' if en else
              'Ниже использованы те же отложенные банки, что и в асимметричной серии ≥50 мс.'), '',
             '| '+('Branch | Validation | Completed | Strict success | Error (deg)' if en else
                    'Ветка | Валидация | Завершено | Строгий успех | Ошибка (°)')+' |',
             '|---|---:|---:|---:|---:|']
    for key, result in results.items():
        for row in result['rows']:
            error = row['late_rmse_all_deg'] if key == 'C' else row['mean_rmse_completed_deg']
            branch = 'asymmetric' if row['branch'] == 'asymmetric_baseline' else row['branch']
            lines.append(f'| {key} / {branch} | {row["validation_fitness"]:.4f} | '
                         f'{row["completed"]}/{row["cases"]} | {row["successes"]}/{row["cases"]} | {error:.3f} |')
    lines += ['', ('For C, error is the mean RMSE over seconds 8–10, retaining the 90° failure penalty. '
                    'For D, it is the mean angle RMSE of completed episodes only. These two error columns '
                    'have different definitions; incomplete runs remain counted in the completion/success columns.' if en else
                    'В C ошибка — средний RMSE за 8–10 с со штрафом 90° при падении. '
                    'В D это средний RMSE угла только завершённых эпизодов. Определения ошибок различаются; '
                    'незавершённые испытания остаются в знаменателях прохождения и успеха.'), '',
              ('Independent confirmation on the new bank, seed 2026102605, fixed before training finished:' if en else
               'Независимое подтверждение на новом банке, seed 2026102605, заданном до окончания обучения:'), '',
              '| '+('Task / variant | Completed | Strict success' if en else
                     'Опыт / вариант | Завершено | Строгий успех')+' |', '|---|---:|---:|']
    for key, result in results.items():
        for variant, row in result['confirmation']['results'].items():
            lines.append(f'| {key} / {variant} | {row["completed"]}/{row["cases"]} | '
                         f'{row["successes"]}/{row["cases"]} |')
    return '\n'.join(lines)


def mirror_figures(results, en):
    """Embed two existing numerical figures with explicit case and panel descriptions."""
    captions = (
        ('Symmetric C, joint: holding +45°. A: phase trajectories for 10×10 initial states '
         '(−50…50°, −1…1 rad/s); 98/100 complete and 81/100 meet strict success. '
         'B: ten initial angles at zero initial velocity. C–F: one example starting at −30°, 0 rad/s: '
         'membrane potentials (last 0.2 s), muscle forces, summed Ia+II+Ib per muscle, and input currents.' if en else
         'Симметричный C, joint: удержание +45°. A: фазовые траектории для сетки 10×10 начальных состояний '
         '(−50…50°, −1…1 рад/с); завершены 98/100, строгий успех 81/100. '
         'B: десять начальных углов при нулевой скорости. C–F: один пример из −30°, 0 рад/с: '
         'мембранные потенциалы (последние 0,2 с), силы мышц, сумма Ia+II+Ib каждой мышцы и входные токи.'),
        ('Symmetric D, C_transfer: the first held-out trajectory, specified before replay. '
         'A: actual and desired phase trajectories; B: actual and desired angle versus time; '
         'C: membrane potentials over the last 0.2 s; D: muscle forces; E: summed afferents; F: applied currents. '
         'The episode completes but fails strict success; the visible lag is included in the reported error. '
         'The phase curve belongs to a system driven by a changing target.' if en else
         'Симметричный D, C_transfer: первая отложенная траектория, выбранная до воспроизведения. '
         'A: фактическая и задаваемая фазовые траектории; B: фактический и целевой углы во времени; '
         'C: мембранные потенциалы за последние 0,2 с; D: силы мышц; E: суммы афферентов; F: входные токи. '
         'Эпизод завершён, но строгий критерий не выполнен; видимое запаздывание включено в ошибку. '
         'Фазовая кривая относится к системе с меняющимся заданием.'),
    )
    figures = []
    for key, experiment, filename, caption in zip(
            ('C', 'D'), ('C_setpoint_balance', 'D_trajectory'),
            ('target_+045.0.png', 'tracking.png'), captions):
        path = EXPERIMENTS/experiment/'results'/MIRROR_STUDY/results[key]['selected']/'figures'/filename
        figures.append(figure(path, caption))
    return '\n\n'.join(figures)


def mirror_commands(results, en):
    """Reconstruct each completed symmetry run from its saved command options."""
    lines = [('Exact commands for the four symmetry branches follow. Paths identify the original '
              'artifacts; use a new --output directory when rerunning. Common option meanings are in §7.1.' if en else
              'Ниже точные команды четырёх симметричных веток. Пути указывают исходные артефакты; '
              'при повторном запуске задайте новый каталог --output. Общий смысл ключей приведён в §7.1.'), '',
             '```bash', 'PY=.venv/bin/python',
             'TRAIN=scripts/AfferentBalance2026/tools/retrain_tensile.py',
             'EX=scripts/AfferentBalance2026/experiments', '```']
    for key, result in results.items():
        for row in result['rows']:
            if 'protocol' not in row:
                continue
            tokens = ['$PY "$TRAIN"']
            for name, value in row['protocol']['options'].items():
                flag = '--'+name.replace('_', '-')
                if isinstance(value, bool):
                    tokens.append(flag if value else '--no-'+name.replace('_', '-'))
                    continue
                for item in value if isinstance(value, list) else [value]:
                    item = str(item).removeprefix(str(REPO)+'/')
                    prefix = 'scripts/AfferentBalance2026/experiments/'
                    quoted = '"$EX"/'+shlex.quote(item[len(prefix):]) if item.startswith(prefix) else shlex.quote(item)
                    tokens.append(flag+' '+quoted)
            lines += ['', f'**{key} / {row["branch"]}**', '', '```bash',
                      ' \\\n  '.join(tokens), '```']
    suffix = '.en' if en else ''
    lines += ['', (f'Pair projection and seed preparation are documented in the [protocol](../experiments/MIRROR_SYMMETRY{suffix}.md). '
                   'The saved input JSON carries `mirror_symmetry=true`; there is no separate training switch. '
                   'Mutation probability applies to an independent pair coordinate. The three published versions are '
                   '`best_genome_fast_muscle.json`, `best_genome_slow_muscle.json` and '
                   '`best_genome_slow_muscle_mirrow.json`. The latter is the symmetric winner in each C/D folder. '
                   'Figure arrays remain beside their PNG/PDF files; rebuilding this article does not rerun training.' if en else
                   f'Проекция пар и подготовка стартов описаны в [протоколе](../experiments/MIRROR_SYMMETRY{suffix}.md). '
                   'Флаг `mirror_symmetry=true` находится в сохранённом входном JSON; отдельного ключа обучения нет. '
                   'Вероятность мутации относится к независимой координате пары. Три опубликованные версии: '
                   '`best_genome_fast_muscle.json`, `best_genome_slow_muscle.json` и '
                   '`best_genome_slow_muscle_mirrow.json`. Последний файл — симметричный победитель в каждой папке C/D. '
                   'Численные массивы рисунков лежат рядом с PNG/PDF; сборка статьи не перезапускает обучение.')]
    return '\n'.join(lines)


def command_section(rows,en):
    """Render exact command-line invocations from saved protocols and explain all options."""
    intro=('Run commands from the repository root using `.venv/bin/python`. Output folders must not already exist. Preserve the source inputs under `results`; they are intentionally not Git artifacts. For a fresh rerun change `--output` to a new directory. `--help` lists syntax, while the table below describes semantics.' if en else
           'Команды выполняются из корня репозитория через `.venv/bin/python`. Каталоги назначения не должны существовать. Исходные JSON в `results` следует сохранять вместе с расчётами; они намеренно не являются Git-артефактами. Для повторения замените `--output` новым каталогом. Синтаксис доступен через `--help`, смысл ключей приведён ниже.')
    explain=[
      ('--experiment','A, B, C or D: selects episode type and genome codec.','A, B, C или D: тип эпизодов и формат генома.'),
      ('--method','joint: all free genes; staged: C table→least squares→polynomial; command_only: D coefficients only.','joint: все свободные гены; staged: таблица→МНК→полином в C; command_only: только коэффициенты D.'),
      ('--seed-genome','Saved winner providing the initial individual and its complete search configuration.','JSON исходной особи вместе с её полной областью поиска.'),
      ('--extra-seed','Repeatable path to an additional initial candidate; it must satisfy the destination bounds.','Повторяемый путь к дополнительной начальной особи; проверяется попадание в границы.'),
      ('--output','New result folder, including source copy, protocols, generations and winner.','Новый каталог результатов: исходный геном, протокол, поколения и победитель.'),
      ('--population','Number of candidate genomes in each evaluated generation.','Число особей в каждом оцениваемом поколении.'),
      ('--generations','Total evaluated generations; staged C divides this budget between its two stages.','Число оценённых поколений; staged делит бюджет между двумя стадиями.'),
      ('--seed','GA random seed; independent episode-bank seeds remain in the saved configuration.','Seed ГА; независимые seed банков условий находятся в сохранённой конфигурации.'),
      ('--workers','Persistent simulation processes; changes parallelism, not mathematical equations.','Число процессов симуляции; определяет параллелизм, а не уравнения.'),
      ('--mutation-probability','Independent, constant probability of mutating each free gene in a child.','Независимая постоянная вероятность мутации каждого свободного гена потомка.'),
      ('--sigma-start / --sigma-end','Initial/final Gaussian standard deviations in normalized coordinates; cosine schedule.','Начальное/конечное σ гауссовых мутаций в нормированных координатах; косинусное расписание.'),
      ('--validation-every','Validate the whole population at this interval, plus first/final generations.','Интервал проверки всей популяции; дополнительно проверяются первое и последнее поколения.'),
      ('--polarity-seeds / --no-polarity-seeds','Enable/disable genetic polarity alternatives. Never changes the physical torque sign.','Включить/выключить генетические варианты полярности. Знак физического момента не меняется.'),
    ]
    lines=[intro,'','| Option | Meaning |','|---|---|']
    for flag,english,russian in explain:lines.append(f'| `{flag}` | {english if en else russian} |')
    lines+=['','```bash','PY=.venv/bin/python','TRAIN=scripts/AfferentBalance2026/tools/retrain_tensile.py','EX=scripts/AfferentBalance2026/experiments','```']
    lines += ['', ('Prepare four admissible starts from a previous winner:' if en else 'Подготовка четырёх допустимых стартов из прежнего победителя:'), '',
              '```bash', '$PY scripts/AfferentBalance2026/tools/prepare_muscle_times.py \\',
              '  --genome path/to/previous_best.json \\',
              '  --config "$EX"/A_balance/genome_cfg.json \\',
              '  --output path/to/new_inputs', '```', '',
              ('Here --genome is the old self-contained source, --config supplies the new bounds and reference times, and --output must be a new directory. The tool creates clipped/reference seeds and their gain-matched variants. C A_transfer additionally screened six target-current scales on training only (seed_training_screen.npz); these six candidate evaluations are additional to the GA totals.' if en else
               'Здесь --genome — прежний самодостаточный геном, --config — новые границы и опорные времена, --output — новый каталог. Сценарий создаёт варианты clipped/reference и их варианты с компенсацией установившихся коэффициентов передачи. Для C A_transfer дополнительно проверены шесть масштабов задающего тока только на обучающем банке (seed_training_screen.npz); эти шесть оценок особей дополняют бюджет ГА.')]
    for row in rows:
        o=row['options'];lines+=['',f'**{row["label"]}**','', '```bash']
        tokens=['$PY "$TRAIN"']
        for key in ('experiment','method','seed_genome','output','population','generations','seed','workers','mutation_probability','sigma_start','sigma_end','validation_every'):
            value=str(row['path'].relative_to(REPO)/'source_genome.json') if key=='seed_genome' else str(o[key])
            value=value.replace('scripts/AfferentBalance2026/experiments/','"$EX"/')
            tokens.append('--'+key.replace('_','-')+' '+value)
        for i,path in enumerate(o['extra_seed']):tokens.append('--extra-seed '+str(row['path'].relative_to(REPO)/'extra_seed_inputs'/f'{i:02d}.json').replace('scripts/AfferentBalance2026/experiments/','"$EX"/'))
        tokens.append('--polarity-seeds' if o['polarity_seeds'] else '--no-polarity-seeds')
        lines.append(' \\\n  '.join(tokens));lines.append('```')
    lines+=['',('Input preparation preserves the old winner separately. New seeds clip muscle times to the domain or use 71/130 ms; gain-matched variants rescale w/P and N to approximate the former steady calcium drive and saturated force. This does not preserve transients. Exact changes are stored in input metadata; all stages use the new bounds.' if en else
               'Прежний победитель сохранён отдельно. Новые старты получены ограничением времён допустимым диапазоном либо установкой 71/130 мс; варианты gain_matched изменяют w/P и N для приближённого сохранения установившегося кальциевого входа и предельной силы. Переходная динамика при этом меняется. Точные изменения перечислены в metadata входов; все стадии используют новые границы.'), '',
      ('To generate phase grids and signals for any completed run:' if en else 'Для фазовых сеток и осциллограмм любого завершённого запуска:'),'','```bash',
      '$PY scripts/AfferentBalance2026/tools/retraining_figures.py \\',
      '  --experiment B_balance \\',
      '  --run "$EX"/B_balance/results/muscle_tau50_20261026/joint','```','',
      ('Here `--experiment` uses the full folder name, and `--run` identifies a completed result folder. Figures always replay the saved winner. They never optimize on the grid.' if en else
       'Здесь `--experiment` — полное имя опыта, `--run` — завершённый каталог результатов. Рисунки строятся по сохранённому победителю; обучение на фазовой сетке не выполняется.'),'',
      '```bash','$PY scripts/AfferentBalance2026/report/build_retraining_article.py',
      '$PY scripts/AfferentBalance2026/report/build_pdf.py \\',
      '  --pandoc /home/nickolai/miniconda3/envs/CPG/bin/pandoc','```','',
      ('The article builder uses completed runs and selects winners by validation; it does not train. The PDF builder typesets only; `--pandoc` specifies the executable. The symmetry series is reported separately in §4.2 and §7.2.' if en else
       'Сборщик статьи читает завершённые запуски и выбирает победителей по проверочной оценке; обучения не запускает. Сборщик PDF только выполняет вёрстку; `--pandoc` задаёт путь к программе. Симметричная серия рассматривается отдельно в §4.2 и §7.2.'),'',
      '| File | Contents |','|---|---|',
      '| `protocol.json`, `source_genome.json` | Exact options, source hash and warm-start input |',
      '| `training/` or `table/`, `polynomial/` | Each GA stage and its independent selection |',
      '| `run.json`, `source_snapshot.zip` | Equation version, source hashes and calculation code |',
      '| `cases.npz` / `banks.json` | Fixed training, validation and test conditions |',
      '| `generations/generation_XXXX.npz` | Every evaluated population, scores, metrics and RNG state |',
      '| `history.csv`, `validation_history.csv` | Training and validation convergence |',
      '| `best_genome.json`, `selection.json` | Validation-selected genome and its generation/index |',
      '| `test_metrics.npz` | Held-out episode results for the frozen winner |',
      '| `figures/*_grid.npz`, `*_zero_speed.npz` | Every phase-grid trajectory and all 10 zero-speed time courses |',
      '| `figures/*_example.npz` | Full 0.1 ms states and currents for the prescribed example |',
      '| `figures/trajectory_tests.npz` | Every D test trajectory, reference and applied current |',
      '| `figures/summary.json` | Independent checks and complete/success counts |',
      '| `report/results/muscle_tau50_20261026/manifest.json` | Selected runs, source hashes and report provenance |',
      '| `explore_genome.ipynb`, D `live_activity.ipynb` | Interactive genome exploration and continuous state visualization |',
    ]
    return '\n'.join(lines)


def appendix(selected,en):
    """List all 52 controller fields, actual winners and experiment-specific bounds."""
    lines=['# '+('Appendix: complete genome' if en else 'Приложение: полный геном')+' {#appendix-parameters}','',
           ('These tables describe the asymmetric ≥50 ms winners in §4.1. Symmetric C/D winners are stored separately in each experiment as best_genome_slow_muscle_mirrow.json (see §4.2 and §7.2). Indices are zero-based. Geometry and both muscle A values are fixed. N is a muscle-unit multiplier. Command coefficients follow the 52 controller fields and are listed separately.' if en else
            'Таблицы относятся к асимметричным победителям ≥50 мс из §4.1. Симметричные победители C/D хранятся отдельно в каждом опыте как best_genome_slow_muscle_mirrow.json (см. §4.2 и §7.2). Индексы начинаются с нуля. Геометрия и оба значения A фиксированы. N — множитель мышечных единиц. Коэффициенты токов следуют после 52 полей контроллера и приведены отдельно.'),'',
           '| Index | Name | A | B | C | D |','|---:|---|---:|---:|---:|---:|']
    items={k:load_candidate(row['experiment'],row['name'],(row['path']/'best_genome.json').resolve()) for k,row in selected.items()}
    first=next(iter(items.values()))
    for i,field in enumerate(first['cfg'].data['parameters']):
        lines.append(f'| {i} | `{field["name"]}` | '+' | '.join(f'{items[k]["physical"][i]:.7g}' if k in items else '—' for k in 'ABCD')+' |')
    for k,item in items.items():
        lines+=['',f'## {k}: '+('bounds and mappings' if en else 'границы и соответствия'),'', '| Index | Parameter meaning / library mapping | Bounds | Scale |','|---:|---|---:|---|']
        for field in item['cfg'].data['parameters']:
            bounds=f'{field["minimum"]:.7g} … {field["maximum"]:.7g}' if field['mutable'] else f'fixed {field["value"]:.7g}'
            lines.append(f'| {field["index"]} | {field["description"]} `{field["system_parameter"]}` | {bounds} | {field.get('scale','fixed')} |')
        if k in 'CD':
            if item['coefficients'].shape[1] == 3:
                lines += ['', 'Quadratic law: c3=0 is shown only for a common table format; it is not an optimized gene.']
            lines+=['','| Neuron | c0 | c1 | c2 | c3 |','|---:|---:|---:|---:|---:|']
            for i,c in enumerate(np.pad(item['coefficients'],((0,0),(0,4-item['coefficients'].shape[1])))):lines.append(f'| {i} | '+' | '.join(f'{x:.9g}' for x in c)+' |')
            lines+=['',('Applied currents at the five training targets:' if en else 'Значения токов на пяти учебных целях:'),'', '| Target (deg) | I0 | I1 |','|---:|---:|---:|']
            for angle in (-45,-22.5,0,22.5,45):
                values=item['coefficients']@np.array([(angle/45)**p for p in range(item['coefficients'].shape[1])])
                lines.append(f'| {angle:g} | {values[0]:.9g} | {values[1]:.9g} |')
    (HERE/('APPENDIX_PARAMETERS.en.md' if en else 'APPENDIX_PARAMETERS.md')).write_text('\n'.join(lines)+'\n')


def main():
    """Generate bilingual reports while refusing missing or unverified final artifacts."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--allow-incomplete',action='store_true',help='Preview completed branches only; never use for final publication.')
    args=parser.parse_args();OUTPUT.mkdir(parents=True,exist_ok=True);configure_fonts(14)
    if not args.allow_incomplete:
        verification = read_json(OUTPUT/'verification.json')
        if verification.get('status') != 'passed':
            raise ValueError('Complete implementation and notebook verification before publication')
    rows=load_runs(args.allow_incomplete);selected=selected_runs(rows)
    mirror, mirror_provenance = mirror_materials()
    for row in selected.values():
        if not (row['path']/'figures/summary.json').exists() and not args.allow_incomplete:raise FileNotFoundError(row['path']/'figures/summary.json')
    scheme=mechanics();analytic=analytics();training=training_plot(rows)
    import importlib.metadata
    versions={name:importlib.metadata.version(name) for name in ('numpy','numba','scipy','matplotlib','nbclient')}
    manifest=dict(model='SpikingNN.Afferents/v3-tensile',preview=args.allow_incomplete,
                  python=sys.version, package_versions=versions,
                  study=STUDY, verification_sha256=digest(OUTPUT/'verification.json'),
                  confirmation_sha256=digest(OUTPUT/'D_confirmation.json'), muscle_time_bounds_ms={'tau_c':[50,100], 'tau_1':[50,200]},
                  selection_rule='Maximum validation fitness within each task and the 50 ms domain.',
                  mirror_symmetry=mirror_provenance,
                  selected={k:dict(run=str(v['path'].relative_to(REPO)),genome_sha256=digest(v['path']/'best_genome.json')) for k,v in selected.items()},
                  runs=[dict(label=r['label'],path=str(r['path'].relative_to(REPO)),options=r['options'],result=r['result']) for r in rows])
    write_json(OUTPUT/'manifest.json',manifest)
    for en in (False,True):
        values=dict(MECHANICS_FIGURE=figure(scheme,('Figure 1. Mechanical geometry and sensory loop. Force arrows show directions, not force scaling.' if en else 'Рисунок 1. Геометрия и сенсорный контур. Стрелки показывают направления сил, а не их масштаб.')),
                    RESULTS=tables(rows,en)+'\n\n'+detailed_results(rows,selected,en)+'\n\n'+timescale_comparison(selected,en),TRAINING_FIGURE=figure(training,('Convergence of every evaluated branch; validation determines selection.' if en else 'Сходимость всех рассчитанных веток; выбор определяется проверочной оценкой.')),
                    COMMANDS=command_section(rows,en), VERIFICATION=verification_text(en),
                    MIRROR_RESULTS=mirror_tables(mirror,en), MIRROR_FIGURES=mirror_figures(mirror,en),
                    MIRROR_COMMANDS=mirror_commands(mirror,en))
        positive=max(analytic['open_loop_eigenvalues']);negative=min(analytic['open_loop_eigenvalues'])
        values['ANALYTICAL_VALUES']=(f'Numerically, J={analytic["inertia"]:.6f} kg·m², Kg={analytic["gravity_coefficient"]:.5f} N·m. Upright open-loop eigenvalues are {positive:.4f} and {negative:.4f} 1/s: the uncontrolled posture is unstable. At ±45°, the required torque magnitude is {abs(analytic["required_torque_Nm"][0]):.5f} N·m and the opposing single-muscle force is {analytic["minimum_single_muscle_force_N"][0]:.3f} N.' if en else
          f'Численно J={analytic["inertia"]:.6f} кг·м², Kg={analytic["gravity_coefficient"]:.5f} Н·м. Собственные значения разомкнутой механики в вертикали: {positive:.4f} и {negative:.4f} 1/s; без управления положение неустойчиво. При ±45° модуль необходимого момента равен {abs(analytic["required_torque_Nm"][0]):.5f} Н·м, а минимальная сила одной противодействующей мышцы — {analytic["minimum_single_muscle_force_N"][0]:.3f} Н.')
        descriptions=[];conclusions=[]
        for key,row in selected.items():
            prefix=f'{key}: {row["label"]}'
            conclusions.append((f'{prefix}: {row["survived"]}/{row["count"]} test episodes completed; {row["success"]}/{row["count"]} met the strict success criterion.' if en else
                                f'{prefix}: завершено {row["survived"]}/{row["count"]} тестовых эпизодов; строгий критерий успеха выполнен в {row["success"]}/{row["count"]}.'))
            folder=row['path']/'figures'
            summary=read_json(folder/'summary.json') if (folder/'summary.json').exists() else None
            if summary and 'grid' in summary:
                successes=sum(x['successes'] for x in summary['grid']);total=sum(x['cases'] for x in summary['grid'])
                conclusions[-1]+=f' Grid: {successes}/{total}.'
            files=sorted(folder.glob('target_*.png')) if key!='D' else [folder/'tracking.png',folder/'all_tracking_errors.png']
            for path in files:
                if not path.exists():continue
                caption=(f'{prefix}. A: 100 initial states; B: 10 zero-speed angle traces; C–F: one fixed example (−30°, 0 rad/s), membrane potentials, forces, summed afferents and applied currents. C shows the final 0.2 s at full temporal resolution.' if key!='D' else
                         f'{prefix}. Continuous tracking with no hidden state resets; failed trajectories, if any, remain included.') if en else (
                         f'{prefix}. A: 100 начальных состояний; B: 10 осциллограмм с нулевой скоростью; C–F: один фиксированный пример (−30°, 0 рад/с), потенциалы, силы, сумма афферентов и входные токи. На C показаны последние 0,2 с с полным временным разрешением.' if key!='D' else
                         f'{prefix}. Непрерывное слежение без скрытых сбросов состояния; неудачные траектории, если есть, включены.')
                descriptions.append(figure(path,caption))
        values['RESULT_FIGURES']='\n\n'.join(descriptions)
        conclusions += [('With exact symmetry, selected C joint completes 407/408 original tests with '
                         '169/408 strict successes; the new confirmation bank gives 408/408 and 166/408. '
                         'Selected D C_transfer completes 15/18 original and 17/18 confirmation trajectories, '
                         'but has 0/18 strict successes on both banks. Symmetry improves static holding in this '
                         'study without solving the dynamic tracking criterion.' if en else
                         'При точной симметрии выбранный C joint завершает 407/408 исходных тестов '
                         'при 169/408 строгих успехах; новый подтверждающий банк даёт 408/408 и 166/408. '
                         'Выбранный D C_transfer завершает 15/18 исходных и 17/18 подтверждающих траекторий, '
                         'но имеет 0/18 строгих успехов на обоих банках. Симметрия улучшила статическое '
                         'удержание в этой серии, однако строгий критерий динамического слежения не достигнут.')]
        values['CONCLUSIONS']='\n\n'.join(conclusions)
        source=(HERE/('article_body.en.txt' if en else 'article_body.ru.txt')).read_text()
        for key,value in values.items():source=source.replace('{{'+key+'}}',value)
        assert '{{' not in source
        (HERE/('REPORT.en.md' if en else 'REPORT.md')).write_text(source)
        appendix(selected,en)
    print(json.dumps(manifest['selected'],indent=2))


if __name__=='__main__':main()
