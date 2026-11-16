# Работа с экспериментами

[English](USER_GUIDE.en.md) · [Настройки](CONFIGURATION.md) · [Команды](COMMANDS.md) · [Результаты](RESULTS.md)

Команды ниже выполняются из `scripts/AfferentBalance2026` активированным Python окружения SpikingNN.
Из корня репозитория: `source .venv/bin/activate`, затем `cd scripts/AfferentBalance2026`.
Можно передать абсолютные пути к `run.py` и конфигу из любого рабочего каталога.
Файлы JSON разрешают свои относительные пути относительно собственного расположения; CLI — относительно cwd.
Эксперимент не устанавливается как отдельный пакет и не добавляет команд библиотеке.

`validate` не интегрирует систему, не компилирует Numba и не создаёт каталог расчёта. `train` без `--config` запрещён.
Все примеры `*_smoke.json` — технические проверки по 0,02 с и 2–3 поколения. Они **не подтверждают качество управления**.
Полные исходные протоколы задают `configs/presets/*_tau50_v1.json`; D всегда требует явного источника.
Все output должны быть новыми. Повторите примеры с другим `--output`, если такие каталоги уже существуют.

## 1. Изменить диапазон и обучить A/B

```bash
python run.py validate --config configs/studies/A_domain_smoke.json --json
python run.py train --config configs/studies/A_domain_smoke.json
python run.py train --config configs/studies/B_smoke.json
```

В коротком исследовании A изменён `genome.overrides.neurons.a.0` (0,01…0,2).
Физическое значение и область поиска разделены: `value` — центр новой популяции, `minimum/maximum/scale` — кодирование.
Для B используйте preset `B_balance_tau50_v1`; ограничения веса `0.02 ≤ w < 0.1` сохраняются.

## 2. Фиксация и освобождение поля

```bash
python run.py validate --config configs/studies/A_fixed_smoke.json --json
python run.py train --config configs/studies/A_fixed_smoke.json
```

`neurons.a.0` фиксирован на 0,1; независимых координат теперь 47.
Для освобождения поля в сохранённом effective config нужно явно указать `mutable: true`, `minimum`, `maximum`, `scale`.
Ниже создаётся новый конфиг, исходный файл остаётся неизменным:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('results/smoke_A_fixed/effective_config.json')
c = json.loads(p.read_text())
c['genome']['overrides'] = {'neurons.a.0': {'mutable': True, 'minimum': 0.005, 'maximum': 0.15, 'scale': 'log'}}
c['storage']['output'] = str(Path('results/smoke_A_free').resolve())
Path('results/free_config.json').write_text(json.dumps(c))
PY
python run.py validate --config results/free_config.json --json
```

Ожидаются 48 координат. Геометрические `a1/a2` по прежней схеме всегда фиксированы. Опечатки, `log` с нижней границей ≤0 и несовпадающие зеркальные границы отклоняются.

## 3. Дообучение в исходном домене

```bash
python run.py train --config configs/studies/A_smoke.json
python run.py finetune --solution results/smoke_A/solution.json --config configs/studies/A_smoke.json --output results/smoke_A_warm
```

Пример задаёт короткий протокол явно. Если убрать `--config`, домен и физика наследуются из решения,
а defaults дообучения — 50 особей, 200 поколений, seed 43, p=0,3, σ 0,2→0,015, validation каждые 20 поколений,
8 процессов и генетические варианты полярности. Это уже полный расчёт.
Поколение 0 участвует в выборе; лучший validation-checkpoint может быть раньше последней популяции.
В `effective_config.json/provenance.sources` записано происхождение каждого значения.

## 4. Несовместимая область и явный prepare

Проверьте решение в узкой области без операции подготовки: в копии `prepare_domain.json`
уберите `initialization.prepare`, затем `validate --solution results/smoke_A/solution.json`.
Если `neurons.a.0` за пределами [0,005; 0,02], диагностика назовёт поле и границы. Никакого clipping при загрузке нет.

```bash
python run.py prepare --solution results/smoke_A/solution.json --config configs/studies/prepare_domain.json --output results/prepared_A
```

Меняется только явно названное поле; аудит находится в `preparation.json`.
Для дообучения подготовленного решения задайте новый короткий config с той же областью, **без** `initialization.prepare`.
Или используйте `finetune --solution results/prepared_A/solution.json --output results/warm_prepared_A` с полными defaults.
Для мышечных времён доступны `prepare.clip_fields`, `prepare.values`, `prepare.gain_match`.
`gain_match` явно разрешает изменение w, N и соответствующей строки P; это приближённое сохранение стационарного усиления, не динамики.

## 5. C joint и staged при одинаковом бюджете

```bash
python run.py train --config configs/studies/C_smoke.json
python run.py train --config configs/studies/C_staged_smoke.json
python run.py compare --runs results/smoke_C results/smoke_C_staged --output results/compare_C
```

Оба примера имеют общий бюджет 2 поколения. Joint оптимизирует сеть и полином вместе.
Staged делит бюджет 1+1: сеть и таблица токов для пяти целей → ограниченный МНК → сеть и полином.
`search.table_generations` задаёт размер первой стадии; остаток выделяется второй. Таблица проверяется на обучающих целевых углах;
полином — также на промежуточных целях. Модель тока зависит только от заданного угла.
Ограничение коэффициента полинома не является ограничением значения суммарного тока.

## 6. Зеркальный геном

```bash
python run.py prepare --solution results/smoke_C/solution.json --config configs/studies/prepare_mirror.json --output results/prepared_C_mirror
python run.py finetune --solution results/prepared_C_mirror/solution.json --config configs/studies/C_mirror_smoke.json --output results/smoke_C_mirror_warm
```

Отражаются обе половины контроллера; `I1(θ)=I0(−θ)`. Для кубического закона размерность 28 вместо 56.
Проекция усредняет физические пары и сохраняет исходный файл. Она сама по себе не доказывает сохранение качества.
В тетрадке допустимы оба названия варианта `slow_muscle_mirror` и историческое `slow_muscle_mirrow`.

## 7. D и непрерывный вход

```bash
python run.py validate --config configs/studies/D_smoke.json --json
python run.py train --config configs/studies/D_smoke.json
python run.py train --config configs/studies/D_compare_smoke.json
```

Источником служит созданное в шаге 5 решение C. D compare проверяет источники и фильтры на validation,
затем сравнивает frozen, command_only и joint. Test используется только после выбора.
В обычной тетрадке или Python:

```python
from scripts.AfferentBalance2026.experiments.D_trajectory.controller import AngleController
controller = AngleController('results/smoke_D/solution.json')
controller.advance(10, duration_s=0.01)
controller.advance(-10, duration_s=0.01)  # Same neuronal, muscle and mechanical state.
```

Для произвольного cwd передайте абсолютный путь. Желаемый угол — команда, а не гарантия её выполнения.

## 8. Воспроизведение выбранного решения

```bash
python run.py replay --solution results/smoke_C/solution.json --output results/replay_C --recording-stride 10
python run.py report --manifest results/replay_C/manifest.json --output results/report_C
```

`solution.json` хранит выбранную особь, даже если она из первого/нулевого поколения.
`stages/*/final_population.npz` — последняя популяция, это другой объект.
Stride меняет только запись, метрики считаются на каждом шаге 0,1 мс.
Отчёт читает переданный manifest и сохранённые траектории; датированная статья в `report/` остаётся историческим материалом.

## 9. Sweep, checkpoint и публикация

```bash
python run.py validate --config configs/studies/sweep_smoke.json --json
python run.py sweep --config configs/studies/sweep_smoke.json
python run.py publish --solution results/smoke_C/solution.json --pointer results/current_C.json
```

`sweep.axes.generations=[2,3]` запускает две независимые линии с разными горизонтами охлаждения мутации.
`search.checkpoints=[1,2]` только проверяет уже идущую линию; GA не перезапускается.
Старый A/B `run_sweep.py --generations 30 60 100` сохраняет именно смысл checkpoints одной линии до 100.
Ни train, ни compare, ни сборка отчёта не обновляют текущий геном. Только явный `publish` создаёт проверяемый указатель.

## Тетрадки и старые данные

В каждом `explore_genome.ipynb` выбирается один `solution`; он используется для сетки, фазовых портретов и сигналов.
В D общий UI находится в `experiments/notebook_ui.py`; шрифты, цвета, окно, начальные условия и выбор файла остались в ячейках.
В `live_activity` сохранено пользовательское окно 0,5 с. Старые outputs и заметки не удалены.
`SPIKINGNN_NOTEBOOK_SMOKE=1` выбирает необученную модель в памяти и не требует старых результатов.
После изменения численного кода перезапустите Jupyter kernel: автоматической замены работающего интегратора нет.

Старые JSON/NPZ читаются явно по версии и сохранённому домену. Их байты не изменяются.
Поле модели `SpikingNN.Afferents/v3-tensile` — идентификатор реализации, не Python import path.
Старое назначение полей 46/47 как токов требует явного `genome.import_legacy`, а не угадывания размера массива.
Исторический runtime запускается отдельно; текущий loader не активирует его автоматически.
