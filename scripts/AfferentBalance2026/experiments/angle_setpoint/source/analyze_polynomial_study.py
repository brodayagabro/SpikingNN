"""Make figures and a report fragment from completed polynomial-search artifacts only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REPO=Path(__file__).resolve().parents[5]
COLORS={2:'#087E8B',3:'#C76335'}
TARGET_COLORS=['#087E8B','#3575BA','#7851A9','#C76335','#54883B']


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'---|'*len(headers),
                      *('| '+' | '.join(map(str,r))+' |' for r in rows)])


def analyze(root):
    root=Path(root).resolve();inputs={}
    def track(path):
        inputs[str(path.relative_to(REPO))]=hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    def read(path):return json.loads(track(path).read_text())
    def rows(path):return list(csv.DictReader(track(path).open()))
    summary=read(root/'summary.json');protocol=read(root/'protocol.json')
    assert summary['protocol']==protocol
    assert protocol['dt_ms']==.1 and protocol['mode']==0 and not protocol['pd_used']
    assert protocol['hyperparameters']['muscle_torque_factor']==10
    for path,expected in protocol['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
    suites=ET.parse(track(root/'tests.xml')).getroot().iter('testsuite')
    audit=[{k:int(s.get(k,0)) for k in ('tests','failures','errors')} for s in suites]
    assert all(s['failures']==s['errors']==0 for s in audit)
    assert sum(s['tests'] for s in audit)==77
    fonts=Path.home()/'.wine/drive_c/windows/Fonts'
    for name in ('times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf'):
        if (fonts/name).exists():font_manager.fontManager.addfont(str(fonts/name))
    plt.rcParams.update({'font.family':'Times New Roman','font.size':18,
        'axes.labelsize':22,'axes.titlesize':23,'xtick.labelsize':18,'ytick.labelsize':18,
        'legend.fontsize':16,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'svg.fonttype':'none'})
    img=ROOT/'img'/root.name;img.mkdir(exist_ok=True,parents=True)
    def save(fig,name):
        for ext in ('png','pdf','svg'):fig.savefig(img/f'{name}.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig)
    run_rows=[];selected={};laws={};metric_rows={};all_traces={}
    for run in summary['runs']:
        path=root/run['directory'];saved=read(path/'result.json')
        assert saved==run
        with np.load(track(path/'final_population.npz'),allow_pickle=False) as d:
            idx=int(d['selected_index']);assert idx==int(np.argmax(d['validation_scores']))
            np.testing.assert_array_equal(d['fixed_controller'],np.load(root/'physical_controller.npy'))
            law=read(path/'current_law.json')
            np.testing.assert_array_equal(d['command_coefficients'][idx],law['coefficients'])
        for bank in ('test','interpolation'):
            m=rows(path/f'{bank}_metrics.csv')
            assert len(m)==run[bank]['cases']
            assert sum(float(r['success']) for r in m)==run[bank]['successes']
        run_rows.append([run['degree'],run['seed'],f"{run['validation']['fitness']:.6f}",
            f"{run['test']['successes']}/{run['test']['cases']}",f"{run['test']['late_rmse_deg']:.4f}",
            f"{run['interpolation']['successes']}/{run['interpolation']['cases']}"])
        if run['directory']==summary['selected_by_degree'][str(run['degree'])]:
            selected[run['degree']]=run;laws[run['degree']]=law
            for bank in ('test','interpolation'):
                metric_rows[run['degree'],bank]=rows(path/f'{bank}_by_target.csv')
                cases=np.genfromtxt(track(root/f'{bank}_cases.csv'),delimiter=',',names=True)
                expected=np.genfromtxt(track(path/f'{bank}_metrics.csv'),delimiter=',',names=True)
                for trace_meta in read(path/f'{bank}_traces.json'):
                    trace_path=root/trace_meta['file'];trial=trace_meta['trial']
                    assert hashlib.sha256(trace_path.read_bytes()).hexdigest()==trace_meta['sha256']
                    with np.load(track(trace_path),allow_pickle=False) as d:
                        tr=d['trace'].copy();case=d['case'];metrics=d['metrics']
                        assert tr.shape[0]==trace_meta['samples'] and float(d['dt_ms'])==.1
                        np.testing.assert_allclose(case,list(cases[trial]),rtol=0,atol=1e-12)
                        np.testing.assert_allclose(metrics,[expected[trial][str(k)] for k in d['metric_columns']],atol=1e-12)
                        np.testing.assert_allclose(tr[:,0],np.arange(len(tr))*.0001,atol=1e-12)
                        np.testing.assert_allclose(tr[0,1:3],case[1:3],atol=1e-12)
                    all_traces[run['degree'],bank,trial]=tr
    fig,ax=plt.subplots(figsize=(10,5.2),layout='constrained')
    for run in summary['runs']:
        h=rows(root/run['directory']/'history.csv')
        ax.plot([int(r['generation']) for r in h],[float(r['best_fitness']) for r in h],
            color=COLORS[run['degree']],ls='-' if run['seed']==protocol['seeds'][0] else '--',lw=2.3,
            label=f"Степень {run['degree']}, seed {run['seed']}")
    ax.set(xlabel='Эпоха',ylabel='Fitness на обучающем банке',ylim=(0,1.02));ax.grid(alpha=.2);ax.legend(frameon=False)
    save(fig,'convergence')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    calibration=rows(root/'calibration_currents.csv');angles=np.linspace(-45,45,181)
    for i,ax in enumerate(axes):
        for degree in (2,3):
            coeff=np.array(laws[degree]['coefficients'])
            currents=np.polynomial.polynomial.polyval(angles/45,coeff[i])
            ax.plot(angles,currents,color=COLORS[degree],lw=2.4,label=f'Степень {degree}')
        ax.scatter([float(r['target_deg']) for r in calibration],
                   [float(r['I_flex' if i==0 else 'I_ext']) for r in calibration],color='#163047',
                   marker='o',s=40,label='Отдельная калибровка')
        ax.set(title='Мотонейрон '+str(i),xlabel='Заданный угол, °',ylabel='Входной ток, ед. модели')
        ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=14)
    save(fig,'current_laws')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for degree,ax in zip((2,3),axes):
        for k,target in enumerate(protocol['targets_deg']):
            tr=all_traces[degree,'test',10*k]
            ax.plot(tr[::10,0],np.rad2deg(tr[::10,1]),lw=2,color=TARGET_COLORS[k],label=f'{target:g}°')
            ax.axhline(target,color=TARGET_COLORS[k],lw=.9,ls=':',alpha=.6)
        ax.set(title=f"Степень {degree}, seed {selected[degree]['seed']}",xlabel='Время, с',ylabel='Угол от вертикали, °',xlim=(0,10))
        ax.grid(alpha=.2)
        ax.legend(frameon=False,ncol=3,fontsize=15,loc='upper center',bbox_to_anchor=(.5,-.25))
    save(fig,'angle_responses')
    overall=next(r for r in summary['runs'] if r['directory']==summary['selected_overall'])
    degree=overall['degree']
    fig,ax=plt.subplots(figsize=(10,5.6),layout='constrained')
    for k,target in enumerate(protocol['targets_deg']):
        tr=all_traces[degree,'test',10*k][::10]
        x=np.rad2deg(tr[:,1]);y=tr[:,2];color=TARGET_COLORS[k]
        ax.plot(x,y,color=color,lw=2,label=f'Цель {target:g}°')
        ax.scatter(x[0],y[0],s=65,facecolors='white',edgecolors=color,zorder=5)
        ax.scatter(x[-1],y[-1],marker='x',s=60,color=color,zorder=6)
        ax.scatter(target,0,marker='*',s=150,color=color,edgecolors='white',linewidth=.5,zorder=7)
        if len(tr)>100:
            for i in (len(tr)//100,len(tr)//50):
                j=min(i+10,len(tr)-1)
                ax.annotate('',xy=(x[j],y[j]),xytext=(x[i],y[i]),
                            arrowprops={'arrowstyle':'-|>','color':color,'lw':1.7})
    ax.set(xlabel='Угол от вертикали θ, °',ylabel='Скорость ω, рад/с')
    ax.grid(alpha=.2)
    ax.legend(frameon=False,ncol=3,fontsize=17,loc='lower center',bbox_to_anchor=(.5,1.02))
    save(fig,'phase_portrait')
    fig,ax=plt.subplots(figsize=(10,3.7),layout='constrained')
    rate=np.array([[int(r['successes'])/int(r['cases']) for r in metric_rows[d,'test']] for d in (2,3)])
    im=ax.imshow(rate,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
    for i,d in enumerate((2,3)):
        for j,r in enumerate(metric_rows[d,'test']):
            ax.text(j,i,f"{r['successes']}/{r['cases']}\n{float(r['late_rmse_deg']):.3f}°",ha='center',va='center',
                    color='white' if rate[i,j]>.6 else '#163047',fontsize=20)
    ax.set(xticks=range(5),xticklabels=[f'{v:g}°' for v in protocol['targets_deg']],
           yticks=[0,1],yticklabels=['Парабола','Кубический'],xlabel='Целевой угол')
    fig.colorbar(im,ax=ax,label='Доля успешных тестов')
    save(fig,'test_success')

    baseline=summary['baseline_static_compensation']
    comparison=table(['Закон','Seed','Val. fitness','Успех test','Поздняя RMSE, °','Успех между целями'],run_rows)
    by_target=table(['Цель, °','Парабола: успех','RMSE, °','Кубический: успех','RMSE, °'],
        [[a['target_deg'],f"{a['successes']}/{a['cases']}",f"{float(a['late_rmse_deg']):.5f}",
          f"{b['successes']}/{b['cases']}",f"{float(b['late_rmse_deg']):.5f}"]
         for a,b in zip(metric_rows[2,'test'],metric_rows[3,'test'])])
    interp=table(['Цель, °','Парабола: успех','RMSE, °','Кубический: успех','RMSE, °'],
        [[a['target_deg'],f"{a['successes']}/{a['cases']}",f"{float(a['late_rmse_deg']):.5f}",
          f"{b['successes']}/{b['cases']}",f"{float(b['late_rmse_deg']):.5f}"]
         for a,b in zip(metric_rows[2,'interpolation'],metric_rows[3,'interpolation'])])
    coefficient_rows=[]
    for d in (2,3):
        for i,values in enumerate(laws[d]['coefficients']):
            coefficient_rows.append([d,i,*[f'{v:.9g}' for v in values],*(['—']*(4-len(values)))])
    coefficients=table(['Степень','Нейрон','c0','c1','c2','c3'],coefficient_rows)
    label='кубический закон' if overall['degree']==3 else 'парабола'
    conclusion=(f"По validation выбран {label}, seed={overall['seed']}. На пяти целевых углах "
        f"получено {overall['test']['successes']}/50 успешных испытаний; средняя поздняя RMSE "
        f"{overall['test']['late_rmse_deg']:.5f}°. На четырёх промежуточных целях — "
        f"{overall['interpolation']['successes']}/32, RMSE {overall['interpolation']['late_rmse_deg']:.5f}°.")
    failed_targets=', '.join(f"{float(r['target_deg']):g}° ({r['successes']}/{r['cases']} успехов)"
                            for r in metric_rows[2,'test'] if int(r['successes'])<int(r['cases']))
    interpretation=(f"У выбранной параболы неполная успешность на основных целях приходится на {failed_targets}. "
                    "Точность между обучающими целями следует оценивать отдельно: выполнение полосы ±3° "
                    "не означает сохранения столь же малой ошибки, как в пяти точках обучения.")
    section=rf'''### Полноценное обучение пяти целям: 15 октября 2026 г.

В новой серии зафиксирован победитель `small_nm_weights` из популяции
50 особей, эпоха 200, seed=42: индекс {protocol['fixed_controller_index']}, выбранный ранее по validation
равновесия. Нейроны, синапсы, мышцы и афферентные параметры больше не изменялись.
Внешний ток зависит только от команды и постоянен внутри каждого эпизода;
текущие угол и скорость в закон токов не подаются. ПД-регулятор не используется.

Сравнивались чистая парабола и кубический полином со свободным членом:

$$
I_i(\theta_*)=\sum_{{k=0}}^d c_{{ik}}\xi^k,\qquad
\xi=\frac{{\theta_*}}{{45^\circ}},\quad d\in\{{2,3\}}.
\tag{{21}}
$$

Здесь c0 — постоянный ток, c1 — линейный, c2 — квадратичный, c3 — кубический
коэффициент каждого мотонейрона. Свободных генов соответственно 6 и 8.
Статический афферентный компенсатор из (20) в этот новый закон не входит.
Ток при нулевой команде теперь разрешено оптимизировать. Масса, геометрия
и остальные гиперпараметры в геном не включены. Все коэффициенты с границами
и пояснениями сохранены в `current_law.json`.

**Пять целей:** −45°, −22.5°, 0°, +22.5°, +45°. На каждую приходится
3 обучающих, 5 validation и 10 test-эпизодов; всего 15/25/50.
Первый эпизод каждой цели начинается с вертикали; остальные — в случайной
окрестности цели ±8°. Начальные скорости распределены в ±0.15 рад/с.
Начиная с третьего эпизода действует толчок ±0.2 рад/с в случайный момент 4–6 с.
Для train, validation и test используются разные фиксированные seed.
Промежуточные цели −33.75°, −11.25°, +11.25°, +33.75° проверены отдельно
в 32 эпизодах и в обучении либо отборе не участвовали.

Сначала на train выполнена отдельная калибровка двух постоянных токов для
каждой цели: популяция 24, 60 эпох, диапазон токов [−1000,1000]. Затем
калибровочные точки аппроксимировались полиномами; это начальное приближение,
а не итоговый закон. Границы каждого коэффициента строятся по минимуму и
максимуму трёх ориентиров (калибровочная аппроксимация, аппроксимация статической
компенсации, ноль) с запасом 200 в каждую сторону.

Основной поиск: **50 особей × 150 эпох**, по seed 42 и 43 для каждой степени.
Использованы общие GA и fitness из разделов 10–11; вероятность мутации
косинусно снижается 0.5 → 0.2, масштаб в нормированных координатах — 0.18 → 0.001.
Четыре основных запуска содержат 30000 оценок «особь × эпоха», калибровка —
7200; повторные неизменные геномы могут обслуживаться кэшем.
Вся конечная популяция каждого запуска оценена на validation. Выбор особи,
seed внутри каждой степени и общей лучшей модели зафиксирован до test.

{comparison}

Контроль на той же сильной сети: прежняя статическая компенсация с нулевым
кубическим остатком даёт {baseline['test']['successes']}/50 на новом test,
позднюю RMSE {baseline['test']['late_rmse_deg']:.5f}°; на промежуточных целях —
{baseline['interpolation']['successes']}/32. Старые 3/76 получены другой слабой
сетью и на другом банке, поэтому напрямую сравнивать эти доли нельзя.

### Качество по отдельным целям

В следующих таблицах для каждой степени показан seed, выбранный по validation:
парабола — {selected[2]['seed']}, кубический закон — {selected[3]['seed']}.
Успех по-прежнему требует входа в ±3° и |ω|≤0.35 рад/с не позже 3 с
с сохранением полосы до 10 с, включая поздний толчок.

{by_target}

Проверка промежуточных углов:

{interp}

Коэффициенты выбранных законов, токи в единицах модели:

{coefficients}

{conclusion}

{interpretation}

![Обучение законов токов](figures_full_afferents/16_setpoint_convergence.png)

**Рисунок 16.** Сходимость четырёх основных поисков при одной fitness-функции.

![Найденные функции токов](figures_full_afferents/17_setpoint_current_laws.png)

**Рисунок 17.** Полиномиальные токи выбранных по validation особей.
Точки — отдельная предварительная калибровка на train, не независимый test.

![Движение к пяти целям](figures_full_afferents/18_setpoint_trajectories.png)

**Рисунок 18.** По одному test-эпизоду со стартом с вертикали для каждой
цели и каждой степени. Пунктиром показаны целевые углы. Метрики в таблицах
учитывают все начальные условия и толчки, а не только эти примеры.

![Фазовые траектории заданного угла](figures_full_afferents/19_setpoint_phase.png)

**Рисунок 19.** Угол и скорость для общей лучшей модели по validation.
Круг — начало, крест — конец, звезда — цель. Это проекция полной динамики
на механические координаты, а не независимая двумерная система.

![Успешность по целям](figures_full_afferents/20_setpoint_success.png)

**Рисунок 20.** Успехи из 10 и средняя поздняя RMSE по каждой целевой позе.
При падении метрика RMSE содержит прежний штраф 90°, как в остальных опытах.

Для обеих выбранных степеней записаны все состояния с шагом 0.1 мс для
двух test-эпизодов на каждую цель: старт с вертикали и старт около цели с толчком.
Так же записаны по два эпизода на каждую промежуточную цель — всего 36 полных
историй. На рисунках прореживается только отображение. Сохраняются также все
конечные популяции, история fitness, контрольные состояния GA, точные банки,
коэффициенты и SHA-256 вычислительных исходников.

Расчёты выполнены с полной моделью Afferents, шагом 0.1 мс и **множителем
мышечного момента 10**. Для сохранения этой версии wrapper загрузил
`Izh_net.py` из проверенного архива в отдельный `runtime/`; текущая рабочая
библиотека с синтаксической ошибкой и изменённым моментом не исправлялась.
Перед поиском прошли **77 тестов**, включая сравнение с библиотечными
классами, отсутствие прямой ошибки цели в динамике и параллельный расчёт.

Результаты относятся к выбранной сети, конечным банкам и горизонту 10 с.
Два seed характеризуют повторяемость этого поиска ограниченно; они не
доказывают глобальную оптимальность полинома или устойчивость для всех углов.
Особенно важно оценивать промежуточные цели отдельно от пяти обучающих.

Источники: [отчёт новой серии](../experiments/angle_setpoint/report/results/{root.name}/REPORT.md),
[численная сводка](../experiments/angle_setpoint/report/results/{root.name}/summary.json),
[коэффициенты общей лучшей модели](../experiments/angle_setpoint/report/results/{root.name}/{overall['directory']}/current_law.json).
'''
    (root/'report_section.md').write_text(section)
    (root/'conclusion.txt').write_text(conclusion+'\n')
    local_section=section.replace('figures_full_afferents/16_setpoint_convergence.png',f'../../../img/{root.name}/convergence.png')
    for number,name in [(17,'current_laws'),(18,'angle_responses'),(19,'phase_portrait'),(20,'test_success')]:
        mapping={17:'current_laws',18:'trajectories',19:'phase',20:'success'}
        local_section=local_section.replace(f'figures_full_afferents/{number}_setpoint_{mapping[number]}.png',f'../../../img/{root.name}/{name}.png')
    local_section=local_section.replace(f'../experiments/angle_setpoint/report/results/{root.name}/','')
    (root/'REPORT.md').write_text('# Удержание пяти целевых углов: парабола и кубический закон\n\n'+local_section)
    verification=dict(status='passed',test_suites=audit,input_sha256=inputs,
        full_traces_verified=len(all_traces),fixed_controller_index=protocol['fixed_controller_index'],
        test_not_used_for_selection=True,model=protocol['model'],dt_ms=.1,
        figures=['convergence','current_laws','angle_responses','phase_portrait','test_success'])
    (root/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n')
    print(conclusion)
    print('Verified full traces:',len(all_traces),'Figures:',img)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    analyze(parser.parse_args().root)
