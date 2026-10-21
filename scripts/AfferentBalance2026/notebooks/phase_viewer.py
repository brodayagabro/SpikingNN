"""Notebook helpers: replay real saved generations, display and export traces."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scripts.AfferentBalance2026.runtime_support import (
    activate_runtime, core_module, current_results, resolve_recorded_path,
    runtime_directory, runtime_package_directory)

load_generation = core_module('training_history').load_generation
replay_generation = core_module('training_history').replay_generation
TRACE_COLUMNS = core_module('dynamics').TRACE_COLUMNS


def configure_fonts():
    """Configure large serif labels, preferring locally installed Times New Roman."""
    folder = Path.home()/'.wine/drive_c/windows/Fonts'
    for name in ('times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf'):
        if (folder/name).exists():
            font_manager.fontManager.addfont(str(folder/name))
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = 'Times New Roman' if 'Times New Roman' in available else 'DejaVu Serif'
    plt.rcParams.update({'font.family': family, 'font.size': 15, 'axes.labelsize': 18,
                         'axes.titlesize': 18, 'legend.fontsize': 11, 'figure.dpi': 110,
                         'mathtext.fontset': 'stix'})
    return family


def catalog(root):
    """Index saved generations by run, combining continued epoch checkpoints and skipping partial writes."""
    result = {}
    for directory in sorted(Path(root).rglob('generations')):
        if 'interrupted' in directory.parts or 'runtime' in directory.parts:
            continue
        run = directory.parent
        if run.name.startswith('epochs_'):
            run = run.parent
        # Also works with the first archived snapshot reader while an atomic
        # generation_XXXX.tmp.npz write is in progress during live training.
        paths=list(run.glob('generations/generation_*.npz'))
        paths+=list(run.glob('epochs_*/generations/generation_*.npz'))
        files={}
        for path in paths:
            suffix=path.stem.removeprefix('generation_')
            if not suffix.isdecimal():continue
            number=int(suffix)
            if number in files:raise ValueError(f'Duplicate generation {number}: {run}')
            files[number]=path
        files=dict(sorted(files.items()))
        if files:
            result[str(run.relative_to(root))] = dict(path=run, generations=files)
    return result


def initial_cases(preset='mixed', target_deg=0.):
    """Build small, large or mixed five-column initial conditions with nonzero velocities."""
    if preset == 'small':
        angles, speeds = [-8,-4,0,4,8], [.15,-.15,.1,.15,-.15]
    elif preset == 'large':
        angles, speeds = [-45,-30,-15,0,15,30,45], [.5,0,-.5,.3,.5,0,-.5]
    elif preset == 'mixed':
        angles, speeds = [-45,-15,-5,0,5,15,45], [.5,-.5,.15,.3,-.15,.5,-.5]
    else:
        raise ValueError(preset)
    return np.array([[np.deg2rad(target_deg),np.deg2rad(a),v,-1.,0.] for a,v in zip(angles,speeds)])


def validation_snapshot(entry):
    """Return the latest completed validation choice as (snapshot, training rank).

    The rank only locates the already chosen individual; it does not reselect it.
    """
    root=entry['path']
    finals=list(root.glob('final_population.npz'))+list(root.glob('epochs_*/final_population.npz'))
    if not finals:
        return None
    candidates=[]
    for path in finals:
        with np.load(path,allow_pickle=False) as data:
            candidates.append((int(data['generation']),path))
    generation,path=max(candidates)
    snapshot=entry['generations'][generation]
    saved=load_generation(snapshot)
    with np.load(path,allow_pickle=False) as final:
        index=int(final['selected_index'])
        assert index==int(np.argmax(final['validation_scores']))
        if 'physical_controllers' in final:
            np.testing.assert_array_equal(saved['physical_controllers'],final['physical_controllers'])
        if str(saved['command_kind'])!='controller':
            np.testing.assert_array_equal(saved['command_coefficients'],final['command_coefficients'])
    order=np.argsort(-saved['scores'],kind='stable')
    rank=int(np.flatnonzero(order==index)[0])+1
    return snapshot,rank


def draw(path, cases, *, ranks=(1,), export=None, time_window_s=None, selection_label='training'):
    """Replay selected training ranks and draw angle-velocity and angle-time trajectories.

    Return (figure, results); optional export writes traces and provenance.
    Every curve comes from simulation with the selected frozen numerical core.
    """
    saved = load_generation(path)
    ranks=tuple(ranks)
    if not ranks:raise ValueError('Choose at least one population rank')
    fig, axes = plt.subplots(1,2,figsize=(15,5.5),layout='constrained')
    all_results=[]
    colors=plt.get_cmap('tab10')
    styles=['-','--',':','-.',(0,(5,1,1,1,1,1))]
    for rank in ranks:
        results=replay_generation(path,cases,rank=rank)
        for i,r in enumerate(results):
            tr=r['trace'];c=r['case'];color=colors(i%10)
            label=rf"$\theta_0$={np.rad2deg(c[1]):g}°, $\omega_0$={c[2]:g}; $\theta_*$={np.rad2deg(c[0]):g}°" if rank==ranks[0] else '_nolegend_'
            linestyle=styles[ranks.index(rank)%len(styles)]
            axes[0].plot(np.rad2deg(tr[:,1]),tr[:,2],color=color,ls=linestyle,lw=1.7,label=label)
            axes[0].scatter(np.rad2deg(tr[0,1]),tr[0,2],color=color,marker='o',s=28)
            axes[0].scatter(np.rad2deg(tr[-1,1]),tr[-1,2],color=color,
                            marker='s' if r['metrics'][5] else 'x',s=45)
            axes[1].plot(tr[:,0],np.rad2deg(tr[:,1]),color=color,ls=linestyle,lw=1.7)
            all_results.append(dict(r,rank=rank))
    for target in np.unique(cases[:,0]):
        axes[0].scatter(np.rad2deg(target),0,marker='*',color='black',s=150,zorder=5)
        axes[1].axhline(np.rad2deg(target),color='black',ls=':',alpha=.4)
    axes[0].set(xlabel='Угол от вертикали θ, °',ylabel='Угловая скорость ω, рад/с',title='Фазовые траектории')
    view_end=float(saved['duration_s']) if time_window_s is None else min(time_window_s,float(saved['duration_s']))
    axes[1].set(xlabel='Время, с',ylabel='Угол от вертикали θ, °',title='Движение конечности',xlim=(0,view_end))
    for ax in axes:ax.grid(alpha=.25)
    axes[0].legend(loc='best')
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[Line2D([],[],color='black',ls=styles[i%len(styles)],
                                  label=f'Training-ранг {rank}') for i,rank in enumerate(ranks)])
    fig.suptitle(f"Поколение {int(saved['generation'])}; выбор по {selection_label}; шаг 0.1 мс; момент без ×10")
    if export:
        export_results(export,path,fig,all_results,selection_label=selection_label)
    return fig,all_results


def metric_table(results):
    """Convert replay results into table rows with angles in degrees and explicit training ranks."""
    return [dict(rank=r['rank'],index=r['population_index'],generation=r['generation'],
                 training_fitness=r['training_fitness'],target_deg=float(np.rad2deg(r['case'][0])),
                 theta0_deg=float(np.rad2deg(r['case'][1])),omega0=float(r['case'][2]),
                 success=bool(r['metrics'][5]),duration_s=float(r['trace'][-1,0]),
                 settling_s=float(r['metrics'][6]),late_rmse_deg=float(np.rad2deg(r['metrics'][7])))
            for r in results]


def export_results(output,path,fig,results,*,selection_label='training'):
    """Save phase PNG/PDF, full traces, metric CSV and source-generation provenance."""
    output=Path(output);output.mkdir(exist_ok=True,parents=True)
    fig.savefig(output/'phase.png',dpi=220,bbox_inches='tight',pad_inches=.25)
    fig.savefig(output/'phase.pdf',bbox_inches='tight',pad_inches=.25)
    for i,r in enumerate(results):
        np.savez_compressed(output/f'trace_{i:03d}.npz',trace=r['trace'],case=r['case'],
                            metrics=r['metrics'],columns=np.array(TRACE_COLUMNS),rank=r['rank'],
                            population_index=r['population_index'])
    rows=metric_table(results)
    with (output/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    (output/'provenance.json').write_text(json.dumps(dict(source=str(Path(path).resolve()),
        sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),mode=0,pd_used=False,
        dt_ms=.1,muscle_torque_factor=1.,selection=selection_label,rows=rows),ensure_ascii=False,indent=2)+'\n')


def compare_generations(files, generations, cases):
    """Replay the training leader at several generations using identical initial cases."""
    fig,axes=plt.subplots(1,len(generations),figsize=(6*len(generations),5),layout='constrained',squeeze=False)
    for ax,number in zip(axes[0],generations):
        results=replay_generation(files[number],cases)
        for r in results:
            tr=r['trace'];case=r['case']
            ax.plot(np.rad2deg(tr[:,1]),tr[:,2],lw=1.5,
                    label=f"{np.rad2deg(case[1]):g}°; {case[2]:g} рад/с")
            ax.plot(np.rad2deg(tr[0,1]),tr[0,2],'o',ms=4)
        ax.scatter(np.rad2deg(np.unique(cases[:,0])),np.zeros(len(np.unique(cases[:,0]))),marker='*',c='k',s=100)
        ax.set(title=f"Поколение {number}, лучший training",xlabel='θ, °',ylabel='ω, рад/с')
        ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=10)
    return fig


def learning_plot(run):
    """Plot best/median training fitness and leader success across a continued run."""
    paths=sorted(Path(run).glob('epochs_*/history.csv')) or [Path(run)/'history.csv']
    with paths[-1].open() as stream:rows=list(csv.DictReader(stream))
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    x=[int(r['generation']) for r in rows]
    for field,label in [('best_fitness','Лучший'),('median_fitness','Медиана')]:
        axes[0].plot(x,[float(r[field]) for r in rows],label=label)
    axes[1].plot(x,[float(r['best_success']) for r in rows],label='Успех лучшего на training')
    axes[0].set(xlabel='Поколение',ylabel='Fitness');axes[0].legend()
    axes[1].set(xlabel='Поколение',ylabel='Доля успешных испытаний',ylim=(-.02,1.02))
    for ax in axes:ax.grid(alpha=.25)
    return fig


def widget_viewer(root):
    """Display interactive controls for run, generation, rank, target and initial conditions."""
    import ipywidgets as widgets
    from IPython.display import display,clear_output
    runs=catalog(root)
    choose=widgets.Dropdown(options=list(runs),description='Обучение:',layout=widgets.Layout(width='95%'))
    generation=widgets.SelectionSlider(options=[1],description='Поколение:',continuous_update=False,layout=widgets.Layout(width='85%'))
    preset=widgets.Dropdown(options=[('Малые ±8°','small'),('Большие ±45°','large'),('Смешанные','mixed')],value='mixed',description='Старт:')
    target=widgets.FloatSlider(min=-45,max=45,step=2.5,value=0,description='Цель, °:',continuous_update=False)
    count=widgets.IntSlider(min=1,max=5,value=1,description='Лучших K:',continuous_update=False)
    window=widgets.FloatSlider(min=.2,max=10.,step=.2,value=1.,description='Окно, с:',continuous_update=False)
    button=widgets.Button(description='Рассчитать траектории',button_style='primary',layout=widgets.Layout(width='240px'))
    output=widgets.Output()
    def changed(*_):
        """Update generation and target controls after a different run is selected."""
        files=runs[choose.value]['generations'];generation.options=list(files);generation.value=max(files)
        data=load_generation(files[max(files)])
        target.disabled=str(data['command_kind'])=='controller'
        if target.disabled:target.value=0
        count.max=min(5,len(data['scores']))
    choose.observe(changed,names='value');changed()
    def render(_):
        """Simulate the chosen candidates and display their phase plots and metrics."""
        with output:
            clear_output(wait=True)
            path=runs[choose.value]['generations'][generation.value]
            fig,results=draw(path,initial_cases(preset.value,target.value),ranks=range(1,count.value+1),time_window_s=window.value)
            display(fig);plt.close(fig)
            try:
                import pandas as pd
                display(pd.DataFrame(metric_table(results)))
            except ImportError:
                print(json.dumps(metric_table(results),ensure_ascii=False,indent=2))
            print('Источник:',path)
    button.on_click(render)
    display(widgets.VBox([choose,generation,widgets.HBox([preset,target]),widgets.HBox([count,window]),button,output]))
    return dict(runs=runs,choose=choose,generation=generation,render=render)
