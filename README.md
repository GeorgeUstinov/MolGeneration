# MolGenerate / MOSTGen

`MolGenerate.ipynb` — каноническая линейная реализация воспроизводимого
скрининга одиночных UV-поглощающих MOST-фотопереключателей. Модульный пакет
`mostgen` внутри notebook остаётся доступен как CLI.

## Что реально работает

- real-data adapters для M13 (включая M11/M12), M01, M05, U07, U09, U12,
  U13 и U16;
- исходный `database_matrix_MOST_UV_skin.xlsx` как каталог источников и
  provenance, но не как молекулярная обучающая таблица;
- отдельные endpoint ensembles: Random Forest reward и Extra Trees evaluator,
  exact-Murcko scaffold split и 90% split-conformal uncertainty;
- три family spaces: NBD/QC, Dewar-pyrimidinone, spiropyran/merocyanine;
- установленный REINVENT4 v4.8.24, pinned commit, официальный LibInvent prior,
  family transfer learning, настоящий staged RL и neural post-sampling;
- exact-library gate без enumerative filler, hard psoralen/furocoumarin veto;
- GFN2-xTB conformer/energy proxy и официальный xtb4stda/sTDA spectrum proxy;
- matched полный reviewer-бюджет и размер result sets, машинный budget ledger,
  diversity/AD/reward diagnostics, review cards, отчёт и пустой лабораторный
  data package. Код не выполняет физические эксперименты.

Доступные данные не содержат достаточной идентифицированной molecular ΔH
таблицы. Поэтому energy reviewer не выдумывается, `ad_most=false`, а итоговый
joint/shortlist закрывается до physical/experimental evidence. λmax-модель
создаёт явно маркированный Gaussian-band proxy, не «полный измеренный спектр».

## Запуск

```bash
./venv/bin/python -m pytest -q
./venv/bin/jupyter nbconvert --to notebook --execute --inplace MolGenerate.ipynb
./venv/bin/python -m mostgen run-all --mode smoke --output runs/smoke
./venv/bin/python -m mostgen run-all --mode production --output runs/production
```

CLI: `prepare-data`, `train-reviewers`, `sample-baselines`, `train-generator`,
`generate`, `review`, `run-all`.

`smoke` применяет только детерминированные synthetic fixtures для быстрых
тестов. `full` и `production` используют real endpoint adapters и реальный
LibInvent. Production config задаёт три seed и не менее 1000 уникальных
структур на запуск; короткий выполненный notebook audit не выдаётся за этот
критерий.

## Границы утверждений

Результаты — исследовательские screening candidates, не безопасные косметические
ингредиенты. ISO 24444/24443 относятся к готовому продукту/формуляции. OECD TG
432 и остальные лабораторные проверки не выполняются программой. Протокол и
формат внешней передачи данных описаны в
[`docs/EXPERIMENTAL_VALIDATION.md`](docs/EXPERIMENTAL_VALIDATION.md).
