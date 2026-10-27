"""Publish validation-selected retrained genomes and annotate population archives."""
import json
import re
from pathlib import Path
import shutil
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
from scripts.AfferentBalance2026.report.build_retraining_article import load_runs,selected_runs,STUDY
from scripts.AfferentBalance2026.experiments.core.genome import file_hash,write_json
from scripts.AfferentBalance2026.experiments.C_setpoint_balance.joint_study import load_joint_genome


def annotate(row):
    """Add loader metadata while proving that every existing numerical array is unchanged."""
    records=[]
    for stage in row['stages']:
        initial=json.loads((stage/'initial_genome.json').read_text())
        schema=initial['schema'];signature=initial['configuration_sha256']
        run=json.loads((stage/'run.json').read_text())
        if 'command' in initial:run['command']=initial['command']
        run['configuration_sha256']=signature
        run['artifact_annotation']='Only loader metadata added after calculation; numerical arrays preserved.'
        write_json(stage/'run.json',run)
        for path in sorted((stage/'generations').glob('generation_*.npz')):
            with np.load(path,allow_pickle=False) as stored:data={k:stored[k] for k in stored.files}
            if 'schema' in data:continue
            before=file_hash(path)
            extended=dict(data,schema=np.array(schema),configuration_sha256=np.array(signature))
            temporary=path.with_suffix('.metadata.npz');np.savez_compressed(temporary,**extended)
            with np.load(temporary,allow_pickle=False) as check:
                for key,value in data.items():np.testing.assert_array_equal(value,check[key])
            temporary.replace(path)
            records.append(dict(file=str(path.relative_to(row['path'])),before=before,after=file_hash(path)))
        last=stage/'generations'/f'generation_{run["generations"]:04d}.npz'
        with np.load(last,allow_pickle=False) as stored:final={k:stored[k] for k in stored.files}
        with np.load(stage/f'validation_{run["generations"]:04d}.npz') as val:
            np.testing.assert_array_equal(final['normalized'],val['normalized'])
            final.update(validation_scores=val['scores'],validation_metrics=val['metrics'])
        np.savez_compressed(stage/'final_population.npz',**final)
    record_path=row['path']/'population_metadata_annotation.json'
    if records:write_json(record_path,dict(numerical_arrays_unchanged=True,files=records))
    # Preserve repeatable additional seeds before canonical winners are updated.
    protocol=json.loads((row['path']/'protocol.json').read_text())
    for i,path in enumerate(row['options']['extra_seed']):
        destination=row['path']/'extra_seed_inputs'/f'{i:02d}.json'
        if destination.exists():continue
        source=REPO/path
        if file_hash(source)!=protocol['extra_seed_sha256'][path]:raise ValueError('Extra seed changed: '+path)
        destination.parent.mkdir(exist_ok=True);shutil.copy2(source,destination)


def publish(key,row):
    """Update convenient winner paths with backups, bilingual indices and notebook notices."""
    run=row['path'];experiment=run.parents[2];results=experiment/'results'
    stats=json.loads((run/'figures/summary.json').read_text())
    if stats['genome_sha256']!=file_hash(run/'best_genome.json'):raise ValueError('Figure genome differs')
    backup=results/STUDY/'previous_current';backup.mkdir(parents=True,exist_ok=True)
    for name in ('best_genome.json','current.json','current_joint.json','summary.json','README.md','README.en.md','best_controller.json','current_law.json'):
        source=results/name
        if source.exists() and not (backup/name).exists():shutil.copy2(source,backup/name)
    shutil.copy2(run/'best_genome.json',results/'best_genome.json')
    pointer=dict(study=STUDY,selected=str(run.relative_to(results)),
                 genome='best_genome.json',best_genome='best_genome.json',model='SpikingNN.Afferents/v3-tensile',
                 selection='validation only',sha256=file_hash(run/'best_genome.json'))
    write_json(results/'current.json',pointer)
    write_json(results/'summary.json',dict(experiment=row['experiment'],selected=row['result'],**stats))
    if key=='C':
        if (experiment/'best_genome.json').exists() and not (backup/'root_best_genome.json').exists():shutil.copy2(experiment/'best_genome.json',backup/'root_best_genome.json')
        shutil.copy2(run/'best_genome.json',experiment/'best_genome.json')
        write_json(results/'current_joint.json',dict(**pointer,study_type='retraining',summary='summary.json',method=row['name']))
        codec,unit=load_joint_genome(run/'best_genome.json');p,c=codec.decode(unit)
        codec.controller.save(results/'best_controller.json',p,source=str(run.relative_to(REPO)))
        write_json(results/'current_law.json',dict(**codec.specification,values=c.tolist(),coefficients=c.tolist()))
    if key=='D':
        source=experiment/'best_genome.json'
        if source.exists() and not (backup/'root_best_genome.json').exists():
            shutil.copy2(source,backup/'root_best_genome.json')
        shutil.copy2(run/'best_genome.json',source)
    result_text=f'{row["survived"]}/{row["count"]}'
    success_text=f'{row["success"]}/{row["count"]}'
    for en in (False,True):
        title='# '+('Retraining with muscle relaxation times ≥50 ms' if en else 'Переобучение с мышечными временами ≥50 мс')
        body=(f'Current winner: `{row["name"]}`. Completed tests: **{result_text}**; strict successes: **{success_text}**. Selected by validation only. Both muscle A parameters are fixed at 0.37. Both muscles use tau_c in [50,100] ms and tau_1 in [50,200] ms. Integration step: 0.1 ms. No PD/PID controller or torque multiplier.' if en else
              f'Актуальный победитель: `{row["name"]}`. Завершено тестов: **{result_text}**; строгий успех: **{success_text}**. Выбор только по проверочному набору. A обеих мышц фиксирован на 0,37. У обеих мышц tau_c ∈ [50,100] мс и tau_1 ∈ [50,200] мс. Шаг 0,1 мс. ПД/ПИД-регулятора и множителя момента нет.')
        paths=(f'\n\n- [Best genome](best_genome.json)\n- [Training result]({run.relative_to(results)}/result.json)\n- [Phase trajectories and signals]({run.relative_to(results)}/figures/)\n- [Scientific article](../../../report/REPORT'+('.en' if en else '')+'.md)\n')
        guide=('All retraining branches are under `muscle_tau50_20261026/`. Each contains its input copy, exact protocol, every evaluated generation, validation-selected genome and held-out test metrics. The figures directory contains PNG/PDF plots and original NPZ arrays. The preceding faster-muscle study remains under tensile_retrain_20261026/; `previous_current/` preserves the former convenience files.' if en else
               'Все ветки находятся в `muscle_tau50_20261026/`. В каждой сохранены копия входа, точный протокол, все поколения, выбранный по проверке геном и независимые тестовые метрики. В figures находятся PNG/PDF и исходные массивы NPZ. Прежняя серия с быстрыми мышцами сохранена в tensile_retrain_20261026/; previous_current содержит прежние файлы-указатели.')
        (results/('README.en.md' if en else 'README.md')).write_text(title+'\n\n'+body+paths+'\n'+guide+'\n')
        readme=experiment/('README.en.md' if en else 'README.md')
        previous=readme.read_text()
        marker='<!-- muscle-tau50-20261026 -->'
        if marker not in previous:
            current=f'{marker}\n\n{title}\n\n{body}\n\n'+('[Current results and file guide](results/README.en.md). [Full article](../../report/REPORT.en.md).' if en else '[Актуальные результаты и описание файлов](results/README.md). [Полная статья](../../report/REPORT.md).')
            history='## Historical documentation (before retraining)' if en else '## Историческая документация до переобучения'
            readme.write_text(current+'\n\n'+history+'\n\n'+previous)
    for notebook in experiment.glob('*.ipynb'):
        if notebook.name not in ('explore_genome.ipynb','live_activity.ipynb'):continue
        data=json.loads(notebook.read_text())
        for cell in data['cells']:
            source=''.join(cell['source'])
            if cell['cell_type']=='markdown':
                # Replace only the previous training notice, preserving user notes.
                notice=(f'<!-- current-training -->\n**Переобучение: времена мышц ≥50 мс.** Геном `{row["name"]}`. Завершено тестов {result_text}, строгий успех {success_text}. Перезапустите ядро. [Результаты](results/README.md).\n\n'
                        f'**English.** Muscle times ≥50 ms; winner `{row["name"]}`. Completed {result_text}; strict success {success_text}. Restart the kernel. [Results](results/README.en.md).\n')
                source=re.sub(r'(?:<!-- current-training -->\n)?\*\*Переобучение.*?\[Results\]\(results/README.en.md\)\.\n?', lambda match: notice, source, flags=re.S)
                source=source.replace('Для текущего генома N0 синий, N1 красный. ', '')
                source=source.replace('The current genome maps N0 to blue and N1 to red, in both voltage and raster views; cross-muscle projections remain possible.', 'Voltage and raster views share these colors; cross-muscle projections remain possible.')
            source=source.replace('Механика v3-tensile: загружен переобученный геном. Метрики проверки приведены в README.', 'Мышечные времена ≥50 мс: загружен переобученный геном. Метрики проверки приведены в README.')
            cell['source']=source.splitlines(keepends=True)
            if cell['cell_type']=='code':cell['outputs']=[];cell['execution_count']=None
        notebook.write_text(json.dumps(data,ensure_ascii=False,indent=1)+'\n')


def main():
    """Finalize all completed branches, then publish four independently verified winners."""
    rows=load_runs(False)
    for row in rows:annotate(row)
    selected=selected_runs(rows)
    for key,row in selected.items():publish(key,row)
    print(json.dumps({k:str(v['path']) for k,v in selected.items()},indent=2))


if __name__=='__main__':main()
