#!/usr/bin/env python3
"""Build five manuscript figures from fresh, validation-selected searches.

Run prepare_fresh_results.py first to generate prescribed replay caches.
No training or simulation is performed by this figure builder.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
from prepare_fresh_results import selected_runs, catalogue, stage_for, CACHE, SOURCE, FRESH

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIG = HERE / 'figures'
SERIES = {'fast': 'expanded fresh sweep', 'slow': 'fresh baseline', 'mirror': 'fresh baseline'}
LABELS = {'fast': 'Fast sweep', 'slow': 'Slow baseline', 'mirror': 'Mirror baseline'}
COLORS = {'fast': '#0072B2', 'slow': '#D55E00', 'mirror': '#009E73'}
PAIR = ('#D55E00', '#0072B2')
EXPS = {'A': 'A_balance', 'B': 'B_balance', 'C': 'C_setpoint_balance'}
FIGURE_TASKS = tuple(EXPS)
INPUTS: dict[str, str] = {}
SELECTED: dict[tuple[str, str], Path] = {}
ROWS: list[dict] = []

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                     'mathtext.fontset': 'dejavuserif', 'font.size': 9,
                     'axes.titlesize': 10, 'axes.labelsize': 9,
                     'legend.fontsize': 7.5, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': .65,
                     'lines.linewidth': 1.2, 'pdf.fonttype': 42,
                     'savefig.dpi': 220})


def record(path):
    path = Path(path)
    key = str(path.relative_to(ROOT))
    if key not in INPUTS:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        INPUTS[key] = h.hexdigest()
    return path


def js(path):
    return json.loads(record(path).read_text())


def npz(path):
    return np.load(record(path), allow_pickle=False)


def csv(path):
    return np.atleast_1d(np.genfromtxt(record(path), names=True, delimiter=','))


def relative_run(path):
    return ROOT / path.split('scripts/AfferentBalance2026/')[-1]


def aggregate(scores):
    return .5 * scores.mean() + .3 * np.sort(scores)[:int(np.ceil(len(scores)/4))].mean() + .2 * scores.min()


def load_selection():
    SELECTED.clear(); ROWS.clear(); INPUTS.clear()
    SELECTED.update(selected_runs())
    for task in EXPS:
        js(FRESH/f'expanded_{task}_selection.json')
    for path in SELECTED.values():
        for name in ('best_genome.json','genome_cfg.json','training.json','provenance.json'):
            record(path/name)
    record(SOURCE/'model.py'); record(SOURCE/'genome.py')
    record(HERE/'prepare_fresh_results.py')
    for (task, group), path in sorted(SELECTED.items()):
        with npz(path / 'test_metrics.npz') as z:
            metrics, score = z['metrics'], float(z['scores'][0])
        assert np.isfinite(metrics).all()
        assert np.isclose(aggregate(metrics[:, 0]), score, atol=1e-12)
        result = js(path / 'result.json')
        genome = js(path / 'best_genome.json')
        complete = metrics[:, 1] >= 10 - 1e-5
        error = metrics[:, 7]
        ROWS.append(dict(task=task, group=group, run=str(path.relative_to(ROOT)),
                         validation=float(result['validation_fitness']), test_fitness=score,
                         cases=len(metrics), completed=int(complete.sum()), success=int(metrics[:, 5].sum()),
                         error_all_deg=float(np.degrees(error).mean()),
                         error_completed_deg=float(np.degrees(error[complete]).mean()) if complete.any() else None,
                         free_coordinates=len(genome['normalized']),
                         selection_generation=result['selected_generation']))
    ROWS.sort(key=lambda r: (r['task'], list(SERIES).index(r['group'])))


def panel(ax, letter, title):
    ax.set_title(f'({letter}) {title}', loc='left', pad=7)
    ax.grid(alpha=.16, lw=.6)
    ax.set_axisbelow(True)


def finish(fig, name):
    # At the publisher's 122 mm text width, keep labels legible after reduction.
    from matplotlib.text import Text
    for label in fig.findobj(match=Text):
        if label.get_text():
            label.set_fontsize(max(9.5, label.get_fontsize()))
    FIG.mkdir(exist_ok=True)
    for suffix in ('pdf', 'png'):
        fig.savefig(FIG / f'{name}.{suffix}', bbox_inches='tight', pad_inches=.07,
                    metadata={'Creator': 'Afferents2026/build_figures.py'} if suffix == 'pdf' else None)
    plt.close(fig)
    print(name, flush=True)


def thin(trace, maximum=6000):
    indices = np.unique(np.r_[np.arange(0, len(trace), max(1, len(trace)//maximum)), len(trace)-1])
    return trace[indices]


def example(task, group, target=0):
    with npz(CACHE / f'{task}_{group}' / f'target_{target:+06.1f}_example.npz') as z:
        return z['trace'], z['metrics'], z['case']


def figure1():
    fig = plt.figure(figsize=(8.2, 6.9), layout='constrained')
    grid = fig.add_gridspec(3, 2, height_ratios=[1.35, 1, 1])
    axes = [fig.add_subplot(grid[i, j]) for i in range(3) for j in range(2)]
    ax = axes[0]; panel(ax, 'a', 'One joint; two tensile muscles')
    theta = np.deg2rad(25); rod = .3*np.array([np.sin(theta), np.cos(theta)]); point = rod/6
    ax.plot([0, rod[0]], [0, rod[1]], color='#333333', lw=4)
    ax.plot([0, 0], [0, .31], '--', color='.7', lw=.8)
    for anchor, color, name in ((np.array([.2, 0]), PAIR[0], 'Flexor'), (np.array([-.2, 0]), PAIR[1], 'Extensor')):
        ax.plot([anchor[0], point[0]], [0, point[1]], color=color, lw=2)
        ax.annotate('', point+.09*(anchor-point)/np.linalg.norm(anchor-point), point,
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
        ax.plot(*anchor, 's', color=color, ms=5); ax.text(anchor[0], -.025, name, ha='center', color=color)
    ax.annotate('', rod/2 + [0, -.08], rod/2, arrowprops=dict(arrowstyle='->', color='#009E73', lw=1.7))
    ax.text(rod[0]/2+.012, rod[1]/2-.05, '$mg$', color='#007658')
    arc = np.linspace(0, theta, 50); ax.plot(.22*np.sin(arc), .22*np.cos(arc), color='.4', lw=.8)
    ax.text(.045, .24, r'$\theta$'); ax.plot(0, 0, 'ko', ms=4)
    ax.text(-.21, .27, '$l=0.30$ m\n$a_1=0.20$ m\n$a_2=0.05$ m', fontsize=8, va='top')
    ax.set(xlim=(-.27, .27), ylim=(-.055, .32), aspect='equal'); ax.axis('off')
    ax = axes[1]; panel(ax, 'b', 'Afferent feedback and command pathway'); ax.axis('off')
    ax.set(xlim=(0, 1), ylim=(0, 1))
    def box(x, y, w, h, label, color='#edf2f6'):
        ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle='round,pad=.012', facecolor=color, edgecolor='.6', lw=.6))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=8)
    box(.02,.75,.27,.17,'Target angle\n'+r'$\theta^*$')
    box(.39,.75,.58,.17,'Command currents\nA/B: 0; C: polynomial')
    box(.60,.42,.37,.19,'2 neurons\n'+r'$v_0,u_0;\ v_1,u_1$', '#e5eef5')
    box(.02,.42,.39,.19,'Antagonist muscles\n'+r'$C\ \to\ x\ \to\ F$', '#fbebdf')
    box(.02,.06,.39,.19,'Limb mechanics\n'+r'$\theta,\omega\ \to\ L,\dot L$')
    box(.60,.06,.37,.19,'Afferent signals\nIa, II, Ib for F/E', '#e1f0e9')
    for start,end in [((.29,.835),(.39,.835)),((.785,.75),(.785,.61)),((.6,.515),(.41,.515)),
                      ((.215,.42),(.215,.25)),((.41,.155),(.60,.155)),((.785,.25),(.785,.42))]:
        ax.annotate('',end,start,arrowprops=dict(arrowstyle='->',lw=1,color='.25'))
    # Force and activation are also sensory inputs, not only geometric length.
    ax.annotate('',(.60,.24),(.41,.43),arrowprops=dict(arrowstyle='->',lw=.8,color='.5'))
    ax.text(.46,.33,'F, x',fontsize=7,color='.4')
    trace, _, _ = example('A','fast')
    zoom = trace[(trace[:,0]>=9.8)&(trace[:,0]<=10)]
    ax=axes[2]; panel(ax,'c','Neuronal activity: resolved spikes')
    for i,color in enumerate(PAIR): ax.plot(zoom[:,0],zoom[:,5+i],color=color,lw=.65,label=f'Neuron {i}')
    ax.set(xlabel='Time (s)',ylabel='Voltage (mV)',xlim=(9.8,10)); ax.legend(ncol=2,loc='lower right')
    t=thin(trace)
    ax=axes[3]; panel(ax,'d','Muscle force')
    for i,color in enumerate(PAIR): ax.plot(t[:,0],t[:,3+i],color=color,lw=.8,label=('Flexor','Extensor')[i])
    ax.set(xlabel='Time (s)',ylabel='Force (N)',xlim=(0,10)); ax.legend(ncol=2)
    ax=axes[4]; panel(ax,'e','Muscle activation')
    for i,color in enumerate(PAIR): ax.plot(t[:,0],t[:,17+i],color=color,lw=.7,label=('Flexor','Extensor')[i])
    ax.set(xlabel='Time (s)',ylabel='Activation x',xlim=(0,10)); ax.legend(ncol=2)
    ax=axes[5]; panel(ax,'f','Separate afferent channels')
    # A common log display keeps Ia/II visible beside the much larger Ib; raw values are unaltered.
    for i,color in enumerate(PAIR):
        for j,style in enumerate(('-', '--', ':')):
            ax.plot(t[:,0],t[:,11+3*i+j],color=color,ls=style,lw=.8,label=f'{("F","E")[i]} / {("Ia","II","Ib")[j]}')
    ax.set(xlabel='Time (s)',ylabel='Signal (model units)',xlim=(0,10)); ax.set_yscale('symlog',linthresh=1)
    ax.set_ylim(bottom=0)
    ax.legend(ncol=3,fontsize=6.7)
    finish(fig,'fig1_methods')


def learning_stages(task,group):
    winner=SELECTED[task,group]
    return [winner/'table',winner/'polynomial'] if (winner/'table').exists() else [winner/'training']


def figure2():
    fig,axes=plt.subplot_mosaic([['A','B'],['C','C']],figsize=(8.2,5.7),layout='constrained',sharey=True)
    for task,letter in zip(FIGURE_TASKS,'abc'):
        ax=axes[task]
        panel(ax,letter,{'A':'A: broad muscle-gain range','B':'B: restricted muscle gain','C':'C: setpoint regulation'}[task])
        for group in SERIES:
            if (task,group) not in SELECTED:continue
            color=COLORS[group]; offset=0
            for n,stage in enumerate(learning_stages(task,group)):
                h=csv(stage/'history.csv');v=csv(stage/'validation_history.csv')
                xs=(offset+h['evaluations'])/1000
                # A stage boundary may change its validation bank; do not draw a fictitious connecting segment.
                ax.plot(xs,h['best_fitness'],color=color,ls=':',lw=.75,alpha=.72)
                count=h['evaluations'][0]
                xv=(offset+v['generation']*count)/1000
                ax.plot(xv,v['selected_fitness'],color=color,lw=1.4,marker='o',ms=2.2,
                        drawstyle='steps-post',label=LABELS[group] if n==0 else None)
                with npz(stage/'baseline_validation.npz') as z:
                    baseline=aggregate(np.asarray(z['metrics']).reshape(-1,9)[:,0])
                ax.scatter([offset/1000],[baseline],s=20,marker='D',facecolor='white',edgecolor=color,zorder=5)
                if n: ax.axvline(offset/1000,color=color,ls='--',lw=.6,alpha=.5)
                offset+=h['evaluations'][-1]
        ax.set(xlabel=r'Training-individual evaluations ($10^3$)',ylabel='Fitness',ylim=(-.025,1.035))
        ax.legend(loc='lower right',fontsize=7)
    fig.legend(handles=[Line2D([],[],color='.3',lw=1.4,marker='o',ms=3,label='Validation incumbent'),
                        Line2D([],[],color='.3',ls=':',lw=1,label='Best training individual'),
                        Line2D([],[],color='.3',ls='',marker='D',mfc='white',ms=4,label='Stage initial controller')],
               loc='outside lower center',ncol=3,frameon=False,fontsize=8)
    finish(fig,'fig2_learning')


def grid_phase(ax,task,group,target):
    with npz(CACHE/f'{task}_{group}'/f'target_{target:+06.1f}_grid.npz') as z:
        cases=z['cases'];metrics=z['metrics']
        norm=matplotlib.colors.Normalize(-50,50)
        for i,case in enumerate(cases):
            t=z[f'trace_{i:03d}'];color=plt.cm.viridis(norm(np.degrees(case[1])))
            ax.plot(np.degrees(t[:,1]),t[:,2],color=color,lw=.42,alpha=.55)
            if metrics[i,1]<10-1e-5: ax.plot(np.degrees(t[-1,1]),t[-1,2],'x',color='#B2182B',ms=4)
        ax.plot(target,0,'*',color='black',ms=9,zorder=8)
        ax.text(.98,.98,f'{int(metrics[:,5].sum())}/100 strict success',ha='right',va='top',transform=ax.transAxes,
                fontsize=8,bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=2))
        ax.set(xlabel=r'Angle $\theta$ (deg)',ylabel=r'Speed $\omega$ (rad/s)')


def zero_traces(ax,task,group,target):
    with npz(CACHE/f'{task}_{group}'/f'target_{target:+06.1f}_zero_speed.npz') as z:
        norm=matplotlib.colors.Normalize(-50,50)
        ax.axhspan(target-3,target+3,color='.92');ax.axhline(target,color='.25',ls='--',lw=.8)
        for i,case in enumerate(z['cases']):
            t=z[f'trace_{i:03d}'];ax.plot(t[:,0],np.degrees(t[:,1]),color=plt.cm.viridis(norm(np.degrees(case[1]))),lw=.85)
        ax.set(xlabel='Time (s)',ylabel=r'Angle $\theta$ (deg)',xlim=(0,10))


def angle_colorbar(fig,axes):
    sm=plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(-50,50),cmap='viridis')
    fig.colorbar(sm,ax=list(axes.flat),label=r'Initial angle $\theta_0$ (deg)',fraction=.025,pad=.02,shrink=.75)


def figure3():
    fig,axes=plt.subplots(2,2,figsize=(8.2,5.5),layout='constrained')
    for j,task in enumerate(('A','B')):
        panel(axes[0,j],'ab'[j],f'{task}: fast-domain recovery, 100 initial states')
        grid_phase(axes[0,j],task,'fast',0)
        panel(axes[1,j],'cd'[j],f'{task}: ten zero-speed starts')
        zero_traces(axes[1,j],task,'fast',0)
    angle_colorbar(fig,axes);finish(fig,'fig3_balance')


def figure4():
    fig,axes=plt.subplots(2,2,figsize=(8.2,5.7),layout='constrained')
    for j,group in enumerate(('fast','mirror')):
        panel(axes[0,j],'ab'[j],f'C: {LABELS[group]}, target +45 deg')
        grid_phase(axes[0,j],'C',group,45)
        panel(axes[1,j],'cd'[j],'Same target: ten zero-speed starts')
        zero_traces(axes[1,j],'C',group,45)
    angle_colorbar(fig,axes);finish(fig,'fig4_setpoints')


def test_cases(path):
    stage=path/'training'
    if not stage.exists():stage=path/'polynomial'
    with npz(stage/'cases.npz') as z:return z['test']


def figure5():
    fig,axes=plt.subplots(2,2,figsize=(8.2,5.8),layout='constrained')
    ax=axes[0,0];panel(ax,'a','Strict success across held-out tests')
    for gi,group in enumerate(SERIES):
        for i,task in enumerate(FIGURE_TASKS):
            row=next((r for r in ROWS if r['task']==task and r['group']==group),None)
            if row is None:continue
            x=i+(gi-1)*.24; rate=100*row['success']/row['cases']
            ax.bar(x,rate,width=.22,color=COLORS[group],label=LABELS[group] if task==('C' if group=='mirror' else 'A') else None)
            ax.text(x,rate+2,str(row['success']),ha='center',fontsize=6.8)
    ax.set(xticks=range(3),xticklabels=['A\nn=96','B\nn=96','C\nn=408'],ylabel='Successful trials (%)',ylim=(0,120))
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=8)
    ax=axes[0,1];panel(ax,'b','C: success at all 17 test targets')
    for group in SERIES:
        path=SELECTED['C',group];cases=test_cases(path)
        with npz(path/'test_metrics.npz') as z:m=z['metrics']
        targets=np.unique(cases[:,0]); rates=[100*m[cases[:,0]==v,5].mean() for v in targets]
        ax.plot(np.degrees(targets),rates,color=COLORS[group],marker='o',ms=3,label=LABELS[group])
    ax.set(xlabel='Target (deg)',ylabel='Successful trials (%)',ylim=(-3,105),xticks=[-45,-22.5,0,22.5,45])
    ax=axes[1,0];panel(ax,'c','C: late error, including failures')
    for group in SERIES:
        path=SELECTED['C',group];cases=test_cases(path)
        with npz(path/'test_metrics.npz') as z:m=z['metrics']
        targets=np.unique(cases[:,0]); errs=[np.degrees(m[cases[:,0]==v,7]).mean() for v in targets]
        ax.plot(np.degrees(targets),errs,color=COLORS[group],marker='o',ms=3,label=LABELS[group])
    ax.axhline(3,color='.45',ls='--',lw=.8);ax.set(xlabel='Target (deg)',ylabel='Mean late RMSE (deg)',yscale='log',xticks=[-45,-22.5,0,22.5,45])
    ax=axes[1,1];panel(ax,'d','C: quadratic command laws')
    targets=np.linspace(-45,45,301)
    for group in ('fast','mirror'):
        g=js(SELECTED['C',group]/'best_genome.json');coef=np.asarray(g['physical'][52:]).reshape(2,3)
        for i,style in enumerate(('-','--')):
            ax.plot(targets,np.polynomial.polynomial.polyval(targets/45,coef[i]),color=COLORS[group],ls=style,label=f'{LABELS[group]} / neuron {i}')
    ax.set(xlabel='Target (deg)',ylabel='Applied current (model units)',xticks=[-45,-22.5,0,22.5,45]);ax.legend(fontsize=6.7)
    finish(fig,'fig5_comparison')


def write_outputs():
    lines=[r'\begin{tabular}{llrrrrr}',r'\toprule',r'Task & Domain & Val. & Complete & Success & Error (deg) & $d$\\',r'\midrule']
    for r in ROWS:
        lines.append(f"{r['task']} & {LABELS[r['group']]} & {r['validation']:.3f} & {r['completed']}/{r['cases']} & {r['success']}/{r['cases']} & {r['error_all_deg']:.3f} & {r['free_coordinates']}"+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (HERE/'results_table.tex').write_text('\n'.join(lines)+'\n')
    catalog=catalogue()
    for row in catalog:
        for name in ('result.json','training.json','provenance.json'):
            record(ROOT/row['run']/name)
    # A short manuscript table plus all 43 complete run records in the supplement.
    lines=[r'\begin{tabular}{@{}llrrr@{}}',r'\toprule',
           r'Task / domain & $N\times G$ & Seed & $p_\mu$ & Selected generation\\',r'\midrule']
    for row in ROWS:
        run=ROOT/row['run'];settings=js(run/'training.json')
        lines.append(f"{row['task']} / {row['group']} & $50\\times{settings['generations']}$ & {settings['seed']} & {settings['mutation_probability']:.1f} & {row['selection_generation']}"+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (HERE/'protocols_table.tex').write_text('\n'.join(lines)+'\n')
    g=js(SELECTED['C','fast']/'best_genome.json');coef=np.asarray(g['physical'][52:]).reshape(2,3)
    lines=[r'\begin{align}']
    for i,c in enumerate(coef):
        law=f'{c[0]:.3f}{c[1]:+.3f}'+r'\xi'+f'{c[2]:+.3f}'+r'\xi^2'
        lines.append(f'I_{i}^C&={law}'+(r',\nonumber\\' if i==0 else r'.\label{eq:currentC}'))
    lines.append(r'\end{align}')
    (HERE/'current_laws.tex').write_text('\n'.join(lines)+'\n')
    for p in CACHE.glob('*/summary.json'):record(p)
    manifest=dict(model='SpikingNN.Afferents/v3-tensile',
                  selection='Maximum saved validation fitness in each fast sweep; slow/mirror are fixed-budget fresh baselines. Test outcomes never select.',
                  initialization='Generic configuration; no archived trained controllers.',
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  package_versions={'numpy':np.__version__,'matplotlib':matplotlib.__version__},
                  article_tasks=list(EXPS),selected_results=ROWS,run_catalog=catalog,
                  example_rule='Prescribed (-30 deg,0) A/fast; all 100 grid starts and all ten zero-speed starts. C target +45 deg fixed before replay; no test-based example selection.',
                  figure_count=5,figure_tasks=FIGURE_TASKS,
                  downsampling='No smoothing. Integration and example voltage 0.1 ms; grids recorded every 2 ms, terminal sample retained.',
                  learning_axis='Evaluated training individuals; staged C includes table and polynomial. Excludes validation and testing.',
                  input_sha256=INPUTS,
                  output_sha256={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(FIG.glob('fig[1-5]_*.pdf'))})
    (HERE/'article_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    protocols=['# Fresh search protocols accompanying Afferents 2026','',
               'All 43 recorded runs start from generic configuration. Seven baseline runs were repeated exactly in an isolated copy; the 36-run expanded grid was executed once. Three fast baseline configurations also occur in the expanded grid, so 43 records represent 40 distinct search configurations, not 43 independent replications.',
               '','Fast grid per task: population 50, generations 200/500/1000, seeds 42/43, mutation probabilities 0.1/0.3. Slow/mirror domains have only the 200-generation baseline. Sigma 0.20 → 0.015; validation every 20 generations. C fast splits the budget equally between table and polynomial stages.',
               '','| Task / domain / run | N × G | Seed | p | Training fitness; success | Validation fitness; success | Test fitness; success |',
               '|---|---:|---:|---:|---:|---:|---:|']
    for r in catalog:
        o=r['options'];result=r['result']
        def cell(bank):
            x=result[bank];return f"{x['fitness']:.6f}; {x['success']}/{x['cases']}"
        protocols.append(f"| {r['task']} / {r['group']} / `{Path(r['run']).relative_to(Path('topubsrc/fresh_results'))}` | {o['population']} × {o['generations']} | {o['seed']} | {o['mutation_probability']} | {cell('training')} | {cell('validation')} | {cell('test')} |")
    protocols += ['','Commands, exact population initialization and selection rules: [standalone README](../../topubsrc/README.md).',
                  '','For an exact rerun from configuration, execute from `topubsrc`:','', '```bash',
                  'python run.py repeat --run fresh_results/expanded_A/G1000/N50_p0.1_seed42 --output results/A_repeat',
                  'python run.py repeat --run fresh_results/expanded_B/G500/N50_p0.3_seed42 --output results/B_repeat',
                  'python run.py repeat --run fresh_results/expanded_C/G1000/N50_p0.3_seed42 --output results/C_repeat','```','']
    (HERE/'SEARCH_PROTOCOLS.md').write_text('\n'.join(protocols)+'\n')
    print('article_manifest.json / results_table.tex',flush=True)


def main():
    load_selection()
    for function in (figure1,figure2,figure3,figure4,figure5):function()
    write_outputs()


if __name__=='__main__':main()
