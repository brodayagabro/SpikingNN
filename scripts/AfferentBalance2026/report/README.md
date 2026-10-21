> 21 октября 2026: численные данные этой исторической серии перенесены в корзину по запросу пользователя. Для повторного построения старых рисунков восстановите их по `archive/cleanup_results_20261021.json`. Новые данные находятся в `experiments/*/results/`.

[English](README.en.md)

# Отчёт по полной модели Afferents без ×10

[REPORT.md](REPORT.md) и [REPORT.pdf](REPORT.pdf) описывают пересчёт от
16 октября 2026 г.: физический мышечный момент без усиления, шаг 0.1 мс,
полная модель Afferents и нейронное управление без ПД.

Серия `results/recompute_unscaled_20261016` содержит 39 историй обучения,
6900 поколений, конечные популяции и полные широкоугольные траектории.
Неудачные испытания включены в результаты.

## Основные файлы

- [Интерактивные рисунки для статьи](article_figures.ipynb): панели A–D,
  выбор сценария, правка оформления и экспорт PNG/PDF/SVG из сохранённых траекторий.
  [Инструкция и протокол](../ARTICLE_FIGURES.md).
- [Итоги](results/recompute_unscaled_20261016/summary.json).
- [Аудит поколений](results/recompute_unscaled_20261016/history_verification.json).
- [Проверка тетрадки](results/recompute_unscaled_20261016/notebook_verification.json).
- [Происхождение отчёта](results/recompute_unscaled_20261016/report_verification.json).
- [Параметры победителей](APPENDIX_PARAMETERS.md).
- [Аналитические оценки управляемости и удержания](ANALYTICAL_CONTROL.md):
  восстановлены из старого отчёта и пересчитаны без ×10; раздел 27 текущего отчёта.
- [Численная проверка аналитических формул](analytical/estimates.json)
  и [код проверки](../analytical_control.py).
- [Тетрадка по поколениям](../notebooks/training_phase_trajectories.ipynb).
- [Карта вычислительных файлов](../archive/pre_ABC_20261020/experiments/RECOMPUTE_UNSCALED.md).

Записи проверки относятся к версии документов на момент сборки. Перевод и
обновление ссылок не меняют численные NPZ/CSV или исходники сохранённого runtime.

## Сборка

```bash
.venv/bin/python scripts/AfferentBalance2026/analyze_unscaled.py
.venv/bin/python scripts/AfferentBalance2026/report/build_pdf.py --pandoc /path/to/pandoc
```

Анализ проверяет сохранённые данные, строит иллюстрации и формирует русский
отчёт. Обучение заново не запускается. `build_pdf.py` выполняет только вёрстку
через Pandoc и LuaLaTeX. Английская Markdown-копия находится рядом с русской;
её PDF отдельно не формируется этим сценарием.

Исторические результаты ×10, старые PDF и готовая презентация перенесены
в корзину. Сохранены исходники построителей, статья в TeX и схемы системы;
для прежней сборки потребуются восстановленные данные.
[Журнал очистки](results/cleanup_factor10_20261017.json).
Аналитические выдержки старого отчёта восстановлены отдельно:
[текст источника](analytical/legacy_excerpt.txt),
[происхождение](analytical/legacy_source.json).

[Фазовые сетки 10×10 и осциллограммы от 19 октября](results/article_phase_grid_20261019/README.md).
