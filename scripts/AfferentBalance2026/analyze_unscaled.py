"""Audit and report the completed factor-1 recomputation."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

REPO=Path(__file__).resolve().parents[2]
BASE=REPO/'scripts/AfferentBalance2026'
sys.path.insert(0, str(REPO))
from scripts.AfferentBalance2026.runtime_support import (
    activate_runtime, core_module, current_results, resolve_recorded_path,
    runtime_directory, runtime_package_directory)



def read(path):
    """Read a JSON result or provenance record."""
    return json.loads(Path(path).read_text())
def table(headers,rows):
    """Format table headers and rows as Markdown."""
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'---|'*len(headers),
                      *('| '+' | '.join(map(str,row))+' |' for row in rows)])+'\n'


def moment_capacity(study,mechanics):
    """Necessary static moment bounds; no claim of closed-loop stability."""
    import numpy as np
    muscles=read(study/'fixed_controller.json')['parameters']['muscles']
    limits={name:p['N']*p['A']*p['tau_1'] for name,p in muscles.items()}
    a1,a2=mechanics.a1_m,mechanics.a2_m
    arm=lambda q:a1*a2*np.sin(q)/np.sqrt(a1*a1+a2*a2-2*a1*a2*np.cos(q))
    rows=[]
    for angle in [-45,-42,-22.5,0,22.5,42,45]:
        q=np.deg2rad(angle)-np.pi/2
        terms=[mechanics.muscle_torque_factor*limits['flexor']*arm(q),
               -mechanics.muscle_torque_factor*limits['extensor']*arm(np.pi-q)]
        low=sum(min(0,x) for x in terms);high=sum(max(0,x) for x in terms)
        required=-mechanics.mass_kg*mechanics.gravity_m_s2*mechanics.length_m/2*np.cos(q)
        rows.append(dict(angle_deg=angle,required_moment_Nm=float(required),available_min_Nm=float(low),
                         available_max_Nm=float(high),feasible=bool(low<=required<=high)))
    result=dict(force_bounds_N=limits,angles=rows,
                assumptions='0 <= activation <= 1; initial force zero; F <= A*N*tau_F; zero speed. Necessary static condition only.')
    (study/'moment_capacity.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def document_report(root):
    """Document current artifacts and record hashes without rerunning dynamics."""
    report=BASE/'report'
    relative=Path(os.path.relpath(root,report)).as_posix()
    # The bilingual navigation is maintained independently from numerical reports.
    # Preserve it when rebuilding the report from a different result directory.
    critical=[root/'manifest.json',root/'history_verification.json',root/'summary.json',
              root/'factor_1/runtime_provenance.json',root/'A1/diagnostics/summary.json',
              root/'angle_setpoint/polynomial/summary.json',root/'angle_setpoint/polynomial/protocol.json',
              root/'angle_setpoint/distant_replay/summary.json',root/'angle_setpoint/compensated_cubic/result.json',
              Path(__file__).resolve(),BASE/'archive/pre_ABC_20261020/experiments/MODEL_METHODS.md']
    for exp in ('equilibrium','small_nm_weights'):
        sweep=root/exp/'sweep';winner=read(sweep/'analysis.json')['best_by_validation']['directory']
        critical.extend([sweep/'analysis.json',sweep/'summary.csv',sweep/'genome_comparison.npz',
                         sweep/winner/'final_population.npz',sweep/winner/'best_genome.json',
                         root/exp/'wide_replay/summary.json',root/exp/'wide_replay/metrics.csv'])
    result=dict(status='passed',muscle_torque_factor=1.,dt_ms=.1,
                input_sha256={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in critical},
                output_sha256={name:hashlib.sha256((report/name).read_bytes()).hexdigest() for name in ('REPORT.md','APPENDIX_PARAMETERS.md')})
    (root/'report_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


def analyze(root):
    """Audit all generation snapshots and rebuild the unscaled result analysis and report."""
    import numpy as np
    manifest=read(root/'manifest.json')
    assert manifest['status']=='completed',manifest['status']
    runtime = activate_runtime(runtime_directory(root))
    sys.path.insert(0,str(BASE/'notebooks'))
    load_generation = core_module('training_history').load_generation
    MECHANICS = core_module('config').MECHANICS
    ControllerDomain = core_module('genome').ControllerDomain
    banks = core_module('cases').banks
    PopulationEvaluator = core_module('population').PopulationEvaluator
    summarize = core_module('polynomial_study').summarize
    fitness = core_module('fitness').fitness
    from phase_viewer import configure_fonts,catalog,compare_generations,initial_cases
    import matplotlib.pyplot as plt
    configure_fonts()
    provenance=read(root/'factor_1/runtime_provenance.json')
    for name,digest in provenance['runtime_sha256'].items():
        assert hashlib.sha256((runtime/name).read_bytes()).hexdigest()==digest,name
    runs=catalog(root);snapshots=0;individuals=0
    for name,entry in runs.items():
        files=entry['generations'];assert list(files)==list(range(1,max(files)+1)),name
        for number,path in files.items():
            d=load_generation(path)
            assert int(d['generation'])==number
            expected=np.array([fitness(m) for m in d['training_metrics']])
            np.testing.assert_array_equal(expected,d['scores'])
            assert int(d['best_index'])==int(np.argmax(d['scores']))
            snapshots+=1;individuals+=len(d['scores'])
        print('History verified:',name,len(files),flush=True)
    assert len(runs)==39 and snapshots==6900 and individuals==325200,(len(runs),snapshots,individuals)
    verification=dict(status='passed',histories=len(runs),generations=snapshots,
                      individuals=individuals,all_training_scores_recomputed=True,
                      mechanics=asdict(MECHANICS),runtime_source_hashes_verified=True)
    (root/'history_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    best={};genomes={};wide={};final={};distances={}
    for exp in ('equilibrium','small_nm_weights'):
        sweep=root/exp/'sweep';analysis=read(sweep/'analysis.json')
        detail_plots = core_module('sweep_detail').main
        argv=sys.argv
        try:
            sys.argv=['sweep_detail','--output',str(sweep)]
            detail_plots()
        finally:
            sys.argv=argv
        assert analysis['completed']==analysis['expected']==60
        assert read(sweep/'verification.json')['status']=='passed'
        best[exp]=analysis['best_by_validation']
        genomes[exp]=read(sweep/best[exp]['directory']/'best_genome.json')
        wide[exp]=read(root/exp/'wide_replay/summary.json')
        with (sweep/'summary.csv').open() as f:final[exp]=[r for r in csv.DictReader(f) if int(r['generations'])==200]
        with np.load(sweep/'genome_comparison.npz') as d:
            selected=np.array(['E=200,' in s for s in d['labels']])
            matrix=d['rmse'][np.ix_(selected,selected)];values=matrix[np.triu_indices(12,1)]
            distances[exp]=[values.min(),np.median(values),values.max()]
    baseline={}
    with PopulationEvaluator(1) as evaluator:
        for exp in best:
            domain=ControllerDomain(exp=='small_nm_weights');p=domain.decode(domain.initial())
            _,metrics=evaluator.evaluate(p[None,:],None,banks(exp)['test'])
            baseline[exp]=summarize(metrics[0])
    (root/'initial_controller_baseline.json').write_text(json.dumps(baseline,indent=2)+'\n')
    poly=root/'angle_setpoint/polynomial'
    shutil.copy2(root/'tests.xml',poly/'tests.xml')
    module=runpy.run_path(str(BASE/'report/analyze_historical_polynomial.py'))
    module['analyze'](poly)
    polynomial=read(poly/'summary.json');selected=next(r for r in polynomial['runs'] if r['directory']==polynomial['selected_overall'])
    capacity=moment_capacity(poly,MECHANICS)
    a1=read(root/'A1/diagnostics/summary.json')
    figures=root/'figures';figures.mkdir(exist_ok=True)
    # Compare matching GA budgets, never differently selected test winners.
    grid_values = core_module('sweep_analysis').grid_values
    grids={}
    for exp in best:
        with (root/exp/'sweep/summary.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        for row in rows:
            for field in ('population','generations'):row[field]=int(row[field])
            for field in ('mutation_probability','validation_fitness'):row[field]=float(row[field])
        grids[exp]=[grid_values(rows,[10,30,50,100],[30,60,100,150,200],p,'validation_fitness') for p in (.1,.3,.6)]
    deltas=[b-a for a,b in zip(grids['equilibrium'],grids['small_nm_weights'])]
    limit=max(abs(d).max() for d in deltas)
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for ax,p,delta in zip(axes,(.1,.3,.6),deltas):
        shown=ax.imshow(delta,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
        ax.set(xticks=range(5),xticklabels=[30,60,100,150,200],yticks=range(4),
               yticklabels=[10,30,50,100],xlabel='Поколение',ylabel='Популяция',title=f'p={p:g}')
    fig.colorbar(shown,ax=axes,label='Validation: малые веса минус обычные')
    fig.savefig(figures/'comparison_delta_validation.png',dpi=180,bbox_inches='tight');plt.close(fig)
    for name in ('A1/seed_43','equilibrium/sweep/mutation_0.300/population_050/seed_42'):
        entry=runs[name]
        fig=compare_generations(entry['generations'],[1,60,200],initial_cases('mixed'))
        fig.savefig(figures/(name.split('/')[0]+'_phase_learning.png'),dpi=180,bbox_inches='tight');plt.close(fig)
    # Historical factor-10 artifacts were explicitly moved to the system trash.
    # Do not recreate them by copying the current factor-1 report under old names.
    report=BASE/'report'
    rel=lambda p:Path(os.path.relpath(p,report)).as_posix()
    image=lambda p,label:f'![{label}]({rel(p)})\n'
    parts=['# Афферентное управление неустойчивым маятником спайковой нейромышечной системой\n',
        'Редакция от 16 октября 2026 г. **Полный пересчёт без множителя ×10.** Полные `Afferents`, шаг 0.1 мс, эпизод 10 с.\n',
        '## 1. Версия и происхождение результатов\n',
        'Мышечный момент равен $F_fh_f-F_eh_e$. Масса и геометрия — фиксированные гиперпараметры. '
        'Общее ядро и `Izh_net.py` согласованы; все обучения и проверки используют нейронный режим 0. '
        'ПД-регулятор не участвует. Старые результаты с ×10 перенесены в системную корзину; '
        'см. [журнал очистки](results/cleanup_factor10_20261017.json). '
        'Настоящий документ ссылается только на новый пересчёт.\n',
        f'Источник: [manifest.json]({rel(root/"manifest.json")}), [точные исходники]({rel(root/"factor_1/source_snapshot.zip")}).\n',
        '## 2. Объём пересчёта\n',
        'Две сетки по 12 линий до 200 поколений (120 контрольных популяций); два дополнительных seed равновесия; '
        'три A1 с обучением на больших углах; пять калибровок токов; четыре полиномиальных поиска; '
        'один поиск компенсированного кубического закона. Всего 39 историй обучения, 6900 сохранённых поколений '
        'и 325200 оценок «особь × поколение» до учёта кэша. Старые архивные протоколы Simple_Afferents не смешиваются с этими сериями.\n']
    methods=(BASE/'archive/pre_ABC_20261020/experiments/MODEL_METHODS.md').read_text()
    from SpikingNN.core.Izh_net import Afferents
    methods=methods.replace('{{RECEPTOR_TABLE}}',table(['Параметр','Значение'],[(k,getattr(Afferents,k)) for k in ('p_v','k_v','k_dI','k_dII','k_nI','k_nII','k_f','const_I','const_II')]))
    muscle=[]
    for exp,g in genomes.items():
        for side,p in g['parameters']['muscles'].items():
            muscle.append([exp,side,*(f'{p[k]:.5g}' for k in ('w','N','A','tau_1')),f"{p['N']*p['A']*p['tau_1']:.5g}"])
    methods=methods.replace('{{MUSCLE_TABLE}}',table(['Опыт','Мышца','w','Nμ','A','τF','ANμτF'],muscle))
    parts.append(methods)
    parts += ['## 12. Банки испытаний и правила отбора\n',
        'В двух основных сетках train/validation/test содержат 12/24/48 эпизодов: θ₀ в ±8°, ω₀ в ±0.15 рад/с; '
        'половина с толчком в 4–6 с величины ±(0.15…0.3) рад/с. Банки совпадают между опытами, seed банка и seed эволюции разделены. '
        'Вся последняя популяция оценивается на validation; затем победитель проверяется на test. Конфигурация также выбирается по validation.\n',
        '## 13. Сетка параметров обучения\n',
        'Размеры популяции 10, 30, 50, 100; постоянная вероятность мутации гена 0.1, 0.3, 0.6; seed=42. '
        'Эпохи 30, 60, 100, 150, 200 — срезы одной продолжающейся линии. Масштаб мутации снижается 0.20→0.015 на полном горизонте 200.\n',
        table(['Опыт','Np','p','Эпоха','Val fitness','Успех test','RMSE, °'],[[e,b['population'],b['mutation_probability'],b['generations'],f"{b['validation_fitness']:.7f}",f"{round(b['test_success']*48)}/48",f"{b['test_late_rmse_deg']:.6f}"] for e,b in best.items()]),
        '## 14. Исходный контроллер до обучения\n',
        table(['Опыт','Успех / 48','Поздняя RMSE, °'],[[e,b['successes'],f"{b['late_rmse_deg']:.5f}"] for e,b in baseline.items()])]
    for i,exp in enumerate(best,15):
        parts += [f'## {i}. {"Равновесие без токов" if exp=="equilibrium" else "Малые нейромышечные веса"}\n',
            table(['Np','p','Val fitness','Успех test','RMSE, °'],[[r['population'],r['mutation_probability'],f"{float(r['validation_fitness']):.7f}",f"{round(float(r['test_success'])*48)}/48",f"{float(r['test_late_rmse_deg']):.5f}"] for r in final[exp]]),
            image(root/exp/'sweep/validation_fitness_detail.png','Validation fitness: различия около единицы'),image(root/exp/'sweep/test_late_rmse_deg.png','Поздняя ошибка test'),
            f'[Полная сводка]({rel(root/exp/"sweep/summary.csv")}); [геном победителя]({rel(root/exp/"sweep"/best[exp]["directory"]/"best_genome.json")}).\n']
    parts += ['## 17. Различия конечных геномов\n',
        image(figures/'comparison_delta_validation.png','Разность validation fitness двух опытов'),
        'RMSE вычислена по 50 канонически нормированным свободным генам; это расстояние параметров, а не физическая ошибка движения. '
        'Ниже минимум/медиана/максимум по 66 парам двенадцати победителей на эпохе 200.\n',
        table(['Опыт','Min','Median','Max'],[[e,*(f'{v:.5f}' for v in d)] for e,d in distances.items()])]
    for exp in best:parts.append(image(root/exp/'sweep/genome_rmse_final.png',f'RMSE геномов {exp}'))
    parts += ['## 18. Проверка на больших отклонениях\n',
        'Для заранее выбранного по validation генома каждого опыта: 19 углов −45…45° с шагом 5°, '
        'три скорости −0.5/0/+0.5 рад/с и три толчка −0.3/0/+0.3 рад/с на 5 с. По 171 испытанию. '
        'Сохранён каждый вычислительный шаг, включая начальное состояние.\n',
        table(['Опыт','Успех','Падения','RMSE, °'],[[e,f"{w['successes']}/{w['cases']}",w['falls'],f"{w['mean_late_rmse_deg']:.5f}"] for e,w in wide.items()])]
    for exp in best:parts.append(image(root/exp/'wide_replay/angle_responses.png',f'Большие углы: {exp}'))
    parts += ['## 19. A1: обучение на больших углах и перенос на малые\n',
        'Популяция 50, 200 поколений, p=0.3, seed 42/43/44. Train: ±15/30/45°, 24 эпизода; '
        'validation 48 и test 96 с |θ₀| в 15…45°, |ω₀|≤0.5. Банки зеркально парные, поэтому эпизоды внутри пары зависимы. '
        'Малые углы не участвуют в обучении и выборе A1. Дополнительно проверены свежие 96 малых стартов.\n',
        'В следующей таблице A1 сравнивается с обучением на малых углах при тех же Np=50, p=0.3 и seed; '
        'это контролируемое сравнение банков обучения, отдельное от победителей полных сеток с Np=100.\n',
        f"Seed A1, выбранный по большой validation: **{a1['selected_seed']}**.\n",
        table(['Обучение / seed','Большие / 96','Сетка / 171','Малые / 48','Свежие / 96','RMSE, °','Сумма сил, Н'],[[r['directory'].replace('equilibrium/sweep/mutation_0.300/population_050/seed_42/epochs_200','small / seed42').replace('equilibrium/replicates/','small / '),r['banks']['large_test']['successes'],r['banks']['wide_grid']['successes'],r['banks']['small_test']['successes'],r['banks']['small_fresh']['successes'],f"{r['banks']['small_fresh']['late_rmse_deg']:.5f}",f"{r['banks']['small_fresh']['mean_total_force_N']:.3f}"] for r in a1['runs']]),
        image(figures/'A1_phase_learning.png','Эволюция фазовых траекторий A1'),
        '## 20. Удержание целевых углов\n',
        'На победителе новой серии малых весов заморожены параметры контроллера и обучены только коэффициенты токов. '
        'Калибровка: 24×60 для каждой из пяти целей. Парабола и кубический полином со свободным членом: 50×150, seed 42/43. '
        'Train/validation/test: 15/25/50; ещё 32 эпизода на промежуточных целях.\n',
        r'$$I_i(\theta_*)=\sum_{k=0}^{d}c_{ik}(\theta_*/45^\circ)^k,\qquad d\in\{2,3\}.$$'+'\n',
        'Токи постоянны внутри эпизода и зависят только от команды; текущие угол и скорость напрямую в этот закон не входят.\n',
        table(['Степень','Seed','Val fitness','Test / 50','RMSE, °','Между / 32'],[[r['degree'],r['seed'],f"{r['validation']['fitness']:.6f}",r['test']['successes'],f"{r['test']['late_rmse_deg']:.5f}",r['interpolation']['successes']] for r in polynomial['runs']]),
        f"Выбран по validation: **{selected['directory']}**. [Подробный отчёт и коэффициенты]({rel(poly/'REPORT.md')}).\n",
        'Верхняя граница силы из модели — ANμτF. Проверка диапазона мышечного момента при нулевой скорости '
        'даёт следующее необходимое статическое условие; его выполнение не гарантирует замкнутую устойчивость.\n',
        table(['Угол, °','Нужный момент','Минимум','Максимум','Допустим'],[[r['angle_deg'],f"{r['required_moment_Nm']:.4f}",f"{r['available_min_Nm']:.4f}",f"{r['available_max_Nm']:.4f}",'да' if r['feasible'] else 'нет'] for r in capacity['angles'] if abs(r['angle_deg'])!=42]),
        'Моменты в Н·м; диапазон рассчитан для независимых сил в пределах 0…ANμτF. Реальная сеть может не реализовывать произвольные сочетания сил.\n',
        image(poly/'figures/angle_responses.png','Движение к пяти целям'),image(poly/'figures/phase_portrait.png','Фазовые траектории удержания углов'),
        f"Далёкие старты, начальная ошибка до 90°: {read(root/'angle_setpoint/distant_replay/summary.json')['summary']['successes']}/5 успехов.\n",
        'Компенсированный кубический закон (прежняя постановка третьего опыта) также обучен заново; '
        f"test: {read(root/'angle_setpoint/compensated_cubic/result.json')['test_success']:.1%}. "
        '[Его результаты](' + rel(root/'angle_setpoint/compensated_cubic/result.json') + '). '
        'Этот запуск использует отдельный банк из 76 test-эпизодов и выбор фиксированной сети на отдельном balance-банке; '
        'его долю успехов нельзя напрямую сравнивать с пятицелевым test из 50 эпизодов.\n',
        '## 21. Истории популяций и тетрадка\n',
        '[training_phase_trajectories.ipynb](../notebooks/training_phase_trajectories.ipynb) читает всю популяцию каждого поколения. '
        'Можно менять поколение, ранг, начальные угол и скорость, целевой угол для закона токов. '
        'Это повторное интегрирование сохранённых решений; интерполяция геномов не применяется. '
        'Рейтинг training отличается от окончательного отбора по validation.\n',
        '## 22. Проверки реализации и результатов\n',
        'До пересчёта прошли 82 теста, включая совпадение ускоренной динамики с библиотекой, ограничения генома, '
        'детерминизм параллельного расчёта, продолжение GA и воспроизведение метрик из снимка поколения. '
        'Проверены хеши замороженного ядра, обе сетки и 6900 снимков: fitness заново вычислена из сохранённых метрик. '
        'Тетрадка проверяется полным выполнением в ядре spikingnn.\n',
        '## 23. Ограничения выводов\n',
        'Основные сетки используют один seed; три seed A1 и два seed полиномов дают ограниченную оценку повторяемости. '
        'Test не использован для выбора победителя. Широкий replay проверяет конечную сетку, а не доказывает глобальную устойчивость. '
        'Fitness не штрафует силу и коактивацию. Поздняя RMSE при падении равна штрафу 90°; её нельзя трактовать как измеренную ошибку установившегося режима. '
        'Превосходство над ПИД и биологическая достоверность этими опытами не установлены.\n',
        '## 24. Где искать данные\n',
        f'Корень серии: `{root.name}` в `report/results/`. `manifest.json` задаёт состав и runtime. '
        '`equilibrium/sweep` и `small_nm_weights/sweep`: сводки, карты, RMSE и все финальные популяции. '
        '`A1`: обучение больших углов и перенос; `angle_setpoint`: полиномы, компенсированный закон и далёкие старты. '
        '`generations/generation_XXXX.npz`: полная популяция, оценки, коэффициенты, механика и хеши ядра. '
        '`history_verification.json`: аудит всех поколений. Результаты вычислений не включаются в Git.\n',
        '## 25. Перспективы\n',
        'Отдельно исследовать энергетическую стоимость и коактивацию; расширить число seed и диапазон начальных скоростей; '
        'оценить устойчивость между целями и под внешними нагрузками. Последующие направления: баланс и удержание позы; '
        'включение регулятора в центральный генератор ритма; несколько конечностей и степеней свободы в графическом движке.\n']
    # Keep additional evaluations attached only to the study they actually
    # evaluated. A new training series must not inherit old supplementary data.
    supplements = report / 'supplements.json'
    if supplements.exists():
        for supplement in read(supplements):
            source = resolve_recorded_path(supplement['source_study'], relative_to=report)
            if source == root.resolve():
                section = (report / supplement['russian']).read_text()
                if section.startswith('[English]'):
                    section = section.split('\n\n', 1)[1]
                parts.append(section)
    (report/'REPORT.md').write_text('\n'.join(parts))
    genes=genomes['equilibrium']['genes'];other=genomes['small_nm_weights']['genes']
    appendix='# Приложение. Параметры новых победителей {#appendix-parameters}\n\n'
    appendix+='Выбраны по validation новых серий без ×10. Масса и геометрия не входят в геном.\n\n'
    appendix+=table(['Индекс','Поле','Опыт 1','Опыт 2'],[[g['index'],g['name'],f"{g['value']:.8g}",f"{h['value']:.8g}"] for g,h in zip(genes,other)])
    appendix+='\n'+table(['Поле','Смысл','Единицы'],[[g['name'],g['description'],g['unit']] for g in genes])
    (report/'APPENDIX_PARAMETERS.md').write_text(appendix)
    (root/'REPORT.md').write_text('# Пересчёт без усиления момента\n\n[Итоговый отчёт](../../REPORT.md).\n\n'+table(['Опыт','Успех test','RMSE, °'],[[e,f"{round(b['test_success']*48)}/48",f"{b['test_late_rmse_deg']:.6f}"] for e,b in best.items()]))
    summary=dict(balance=best,wide=wide,A1_selected_seed=a1['selected_seed'],polynomial_selected=selected,
                 history_audit=verification,report=str(report/'REPORT.md'))
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    document_report(root)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE/'report/results/recompute_unscaled_20261016')
    analyze(parser.parse_args().output.resolve())
