#!/usr/bin/env python3
"""Build the self-contained, linear GOSHA.ipynb from the audited source tree."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "GOSHA.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": source.strip() + "\n",
    }


cells: list[dict] = []
cells.append(markdown(r"""
# Воспроизводимый генератор UV-поглощающих MOST-молекул

Самодостаточная линейная реализация и аудит работоспособности

Этот notebook содержит **исходный код всей Python-реализации**, конфигурацию,
последовательный запуск всех стадий и проверку критериев приёмки. Ячейки следует
выполнять сверху вниз (`Restart Kernel and Run All`). По умолчанию выбран
`smoke`-режим: три метода, один seed и 40 молекул на метод. Для полного
эксперимента достаточно заменить `RUN_MODE = "smoke"` на `"full"`; проверенный
full-прогон занимает около пяти минут на текущем CPU.

## Короткий ответ: работает ли генерация?

**Программно — да.** Полный прогон уже сформировал 9000 строк: 3 метода × 3
seed × 1000 валидных уникальных структур. Машинная acceptance-проверка пройдена,
псораленовых/фурокумариновых ядер в `generated.csv` нет. Все 38 тестов проходят.

**Как научно доказательный генератор — пока нет.** Рабочий CPU backend является
reaction-library enumeration + adaptive search, а не обученной нейросетью
LibInvent. Его reviewer-метки синтетические и служат только для проверки
алгоритма. REINVENT4/LibInvent оформлен production TOML-конфигурациями и
ExternalProcess bridge, но prior и сам REINVENT не установлены. GFN2-xTB и
sTDA-xTB также отсутствуют, поэтому 11 прошедших независимый evaluator структур
остались provisional, а итоговый `selected` закономерно равен нулю.
"""))

cells.append(markdown(r"""
## 1. Среда и проверенный full-результат

Ниже фиксируется рабочая директория и версии библиотек. Notebook не импортирует
локальный пакет `mostgen`: в разделе 3 каждый модуль загружается из исходника,
встроенного непосредственно в `.ipynb`. Файлы в `runs/full` используются только
для показа уже выполненной acceptance-проверки и не нужны smoke-запуску.
"""))

cells.append(code(r"""
from pathlib import Path
import importlib.metadata
import json
import os
import platform
import sys
from IPython.display import display, Markdown
from rdkit import RDLogger

# Enumeration intentionally rejects many chemically invalid combinations.  RDKit
# reports every rejected intermediate through its logger; the validity counters
# below retain that information without flooding the persisted notebook output.
for channel in ("rdApp.debug", "rdApp.info", "rdApp.warning", "rdApp.error"):
    RDLogger.DisableLog(channel)

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "config" / "default.json").is_file():
    raise RuntimeError("Запускайте GOSHA.ipynb из директории TEST")

required = ["rdkit", "numpy", "pandas", "scikit-learn", "PyYAML"]
versions = {}
for package in required:
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"Не установлен {package}. Выполните: ./venv/bin/pip install -r requirements.txt"
        ) from exc

print("project:", PROJECT_ROOT)
print("python:", sys.version.split()[0])
print("platform:", platform.platform())
print("packages:", versions)
"""))

cells.append(code(r"""
FULL_RESULT_DIR = PROJECT_ROOT / "runs" / "full"
if (FULL_RESULT_DIR / "verification.json").is_file():
    full_verification = json.loads((FULL_RESULT_DIR / "verification.json").read_text())
    full_review = json.loads((FULL_RESULT_DIR / "review" / "review_summary.json").read_text())
    display({
        "acceptance_passed": full_verification["passed"],
        "generated_rows": full_verification["generated_rows"],
        "run_counts": full_verification["run_counts"],
        "zero_psoralen_cores": full_verification["checks"]["zero_psoralen_cores"],
        "independent_reviewed": full_review["reviewed"],
        "provisional": full_review["independent_joint_pass"],
        "selected": full_review["selected_after_physical_oracle"],
        "physical_oracle": full_review["physical_oracle"]["status"],
    })
else:
    print("Предварительный full-прогон не найден; notebook остаётся полностью исполнимым.")
"""))

cells.append(markdown(r"""
## 2. Архитектура по стадиям

1. **Prepare data.** RDKit канонизирует структуры, строит ground/charged пары с
   одинаковой формулой, удаляет дубликаты, формирует Murcko scaffold split и
   записывает provenance. Встроенные значения — честно помеченные
   `synthetic_smoke_only` fixtures.
2. **Train reviewers.** Reward ensemble (`RandomForestRegressor`) и отдельный
   evaluator ensemble (`ExtraTreesRegressor`) обучаются раздельно. Spectrum
   предсказывается на сетке 290–400 нм; MOST-регрессии разделены по семействам.
3. **Prepare generator.** Для NBD/QC, Dewar-pyrimidinone и spiropyran создаются
   отдельные REINVENT4 v4.8 LibInvent TOML, scaffold SMILES и ExternalProcess
   scoring bridge. CPU run использует то же дискретное пространство синтонов.
4. **Matched-budget search.** Сравниваются random prior, weighted retraining и
   curriculum policy. Каждый метод получает одинаковое число reviewer-вызовов.
5. **Metrics.** Считаются validity, uniqueness, novelty, diversity, AD coverage,
   joint success, bootstrap intervals, reward ESS и абляции.
6. **Independent review.** Top-50–100 переоцениваются evaluator-моделями,
   создаются карточки и XYZ-очередь GFN2-xTB/sTDA-xTB. При отсутствии oracle
   итоговый shortlist закрывается (`selected=false`).
7. **Report and verification.** Формируются отчёт, 7-минутная презентация,
   machine-readable acceptance и SHA-256 manifest.

ITI не генерируется и оставлен только как class-shift validation family.
"""))

cells.append(markdown(r"""
## 3. Встроенная реализация

Следующая служебная ячейка создаёт из встроенных ниже исходников модули только в
памяти текущего kernel. Это сохраняет обычную модульную структуру и относительные
импорты, одновременно делая notebook самодостаточным и обозримым линейно.
"""))

cells.append(code(r"""
import types

for loaded_name in list(sys.modules):
    if loaded_name == "mostgen" or loaded_name.startswith("mostgen."):
        del sys.modules[loaded_name]

embedded_package = types.ModuleType("mostgen")
embedded_package.__package__ = "mostgen"
embedded_package.__path__ = [str(PROJECT_ROOT / "mostgen")]
embedded_package.__file__ = str(PROJECT_ROOT / "mostgen" / "__init__.py")
sys.modules["mostgen"] = embedded_package

def _load_embedded_module(name: str, source: str, filename: Path):
    module = types.ModuleType(name)
    module.__file__ = str(filename)
    module.__package__ = name.rpartition(".")[0]
    sys.modules[name] = module
    exec(compile(source, str(filename), "exec"), module.__dict__)
    return module

print("Изолированный in-memory package namespace подготовлен")
"""))

module_order = [
    "__init__.py", "config.py", "provenance.py", "chemistry.py", "numerics.py",
    "data.py", "reviewers.py", "scoring.py", "search.py", "metrics.py",
    "oracle.py", "review.py", "reporting.py", "validation.py", "cli.py", "__main__.py",
]
descriptions = {
    "__init__.py": "Версия пакета",
    "config.py": "Конфигурация, режимы и контроль бюджета",
    "provenance.py": "SHA-256, версии, manifest и event log",
    "chemistry.py": "RDKit-канонизация, изомерные пары, ECFP/Murcko и safety veto",
    "numerics.py": "AUC, critical wavelength, Beer–Lambert и transforms",
    "data.py": "Reaction library, fixture labels, scaffold split и provenance",
    "reviewers.py": "Reward/evaluator ensembles и applicability domains",
    "scoring.py": "Прозрачная карточка прогноза, pass gates и dense reward",
    "search.py": "Три matched-budget стратегии и REINVENT4 manifests",
    "metrics.py": "Diversity, bootstrap, ESS и абляции",
    "oracle.py": "Конформеры и очередь независимого GFN2/sTDA-xTB oracle",
    "review.py": "Независимая top-переоценка и fail-closed shortlist",
    "reporting.py": "Научный отчёт и материал 7-минутной презентации",
    "validation.py": "Машинная проверка критериев приёмки",
    "cli.py": "Единый CLI и run-all orchestration",
    "__main__.py": "Запуск через python -m mostgen",
}

for filename in module_order:
    source = (ROOT / "mostgen" / filename).read_text(encoding="utf-8")
    if filename == "__init__.py":
        cell_source = (
            "_init_source = r'''" + source + "'''\n"
            "exec(compile(_init_source, embedded_package.__file__, 'exec'), embedded_package.__dict__)\n"
            "print('loaded mostgen', embedded_package.__version__)"
        )
    else:
        module_name = "mostgen." + filename[:-3]
        cell_source = (
            f"_source = r'''{source}'''\n"
            f"_load_embedded_module('{module_name}', _source, PROJECT_ROOT / 'mostgen' / '{filename}')\n"
            f"print('loaded {module_name}')"
        )
    cells.append(markdown(f"### 3.{len(cells) - 6}. {descriptions[filename]} — `{filename}`"))
    cells.append(code(cell_source))

external_source = (ROOT / "scripts" / "reinvent_external_score.py").read_text(encoding="utf-8")
cells.append(markdown(r"""
### ExternalProcess bridge для настоящего REINVENT4

Этот исходник принимает SMILES через stdin в официальном ExternalProcess
формате REINVENT4 и возвращает JSON payload. В notebook он хранится как строка,
поскольку отдельный REINVENT-процесс должен запускать его как исполняемый файл.
"""))
cells.append(code("REINVENT_EXTERNAL_SCORER_SOURCE = r'''" + external_source + "'''\nprint('embedded external scorer lines:', len(REINVENT_EXTERNAL_SCORER_SOURCE.splitlines()))"))

config_text = (ROOT / "config" / "default.json").read_text(encoding="utf-8")
cells.append(markdown(r"""
## 4. Разрешённое пространство и воспроизводимая конфигурация

Конфигурация тоже встроена в notebook. `smoke` меняет только число rows/seeds и
не ослабляет химические либо safety-ограничения. Для production требуются
реальные datasets, checksum prior, REINVENT executable и физический toolchain.
"""))
cells.append(code(
    "CONFIG_JSON = r'''" + config_text + "'''\n"
    "from mostgen.config import validate_config\n"
    "RUN_MODE = 'smoke'  # замените на 'full' для 3 × 3 × 1000\n"
    "config = json.loads(CONFIG_JSON)\n"
    "config['_config_path'] = str(PROJECT_ROOT / 'config' / 'default.json')\n"
    "if RUN_MODE == 'smoke':\n"
    "    smoke = dict(config['smoke'])\n"
    "    config['execution'].update({k: v for k, v in smoke.items() if k != 'training_rows_per_family'})\n"
    "    config['execution']['mode'] = 'smoke'\n"
    "    config['training_rows_per_family'] = smoke['training_rows_per_family']\n"
    "else:\n"
    "    config['execution']['mode'] = 'full'\n"
    "    config['training_rows_per_family'] = 180\n"
    "validate_config(config)\n"
    "display({k: config['execution'][k] for k in ['mode', 'methods', 'seeds', 'n_per_run', 'reviewer_budget_per_run', 'shortlist_size']})"
))

cells.append(markdown(r"""
## 5. Как формируется оценка

Для предсказанного спектра $A(\lambda)$ вычисляются

$$\mathrm{{AUC}}_{{UVB}}=\int_{{290}}^{{320}}A(\lambda)d\lambda,\qquad
\mathrm{{AUC}}_{{UVA}}=\int_{{320}}^{{400}}A(\lambda)d\lambda.$$

$\lambda_c$ — длина волны, на которой накоплено 90% площади 290–400 нм.
Beer–Lambert proxy использует фиксированную условную загрузку:
$T(\lambda)=10^{{-A(\lambda)}}$. Для каждого ensemble endpoint сохраняются
среднее $\mu$, разброс $\sigma$ и нижняя граница
$LCB=\mu-1.645\sigma$.

Непрерывные компоненты $s_i\in[0,1]$ получаются sigmoid/interval transforms.
Итоговая плотная награда:

$$R=\exp\left(\frac{{\sum_i w_i\log(\max(10^{{-3}},s_i))}}{{\sum_i w_i}}\right).$$

После этого повторный ECFP-кластер получает штраф
$R'=R/(1+0.12n_{{cluster}})$. Hard veto (псорален/фурокумарин, известная
фототоксичная структура, reactive alert, неподдерживаемый элемент, неправильная
валентность или невозможная пара) применяется **до** reward.

Joint pass требует одновременно: UVB и UVA LCB не хуже семейных медиан;
$\lambda_c$ LCB ≥ 370 нм; energy LCB положительна и не хуже семейной медианы;
$t_{{1/2}}(305K)$ находится в 4–24 ч; обе AD пройдены; phototoxicity не
неопределённа и её UCB ≤ 0.35; Kp UCB ≤ −4; SA proxy ≤ 5.
"""))

cells.append(markdown(r"""
## 6. Линейный запуск всех стадий

Результаты notebook записываются отдельно в `runs/notebook-smoke` или
`runs/notebook-full`, поэтому проверенный `runs/full` не изменяется.
"""))

cells.append(code(r"""
from mostgen.config import dump_resolved_config
from mostgen.provenance import append_event, write_manifest

OUTPUT_DIR = PROJECT_ROOT / "runs" / f"notebook-{RUN_MODE}"
DATA_DIR = OUTPUT_DIR / "data"
MODELS_DIR = OUTPUT_DIR / "models"
GENERATOR_DIR = OUTPUT_DIR / "generator"
METRICS_DIR = OUTPUT_DIR / "metrics"
REVIEW_DIR = OUTPUT_DIR / "review"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
dump_resolved_config(config, OUTPUT_DIR / "config.resolved.json")
source_inputs = [
    PROJECT_ROOT / "GOSHA.ipynb", PROJECT_ROOT / "config" / "default.json",
    PROJECT_ROOT / "database_matrix_MOST_UV_skin.xlsx",
    PROJECT_ROOT / "Задание.docx",
    PROJECT_ROOT / "Солнцезащитная плёнка с молекулярным накоплением солнечной энергии.pptx",
]
write_manifest(OUTPUT_DIR / "experiment_manifest.json", config, source_inputs, ["GOSHA.ipynb", "Run All"])
print("output:", OUTPUT_DIR)
"""))

cells.append(markdown("### Стадия 1 — подготовка данных и reaction library"))
cells.append(code(r"""
from mostgen.data import prepare_data, read_csv

data_result = prepare_data(config, DATA_DIR, PROJECT_ROOT)
append_event(OUTPUT_DIR / "events.jsonl", "prepare-data", data_result)
display({k: data_result[k] for k in ["training_rows", "library_rows", "raw_inputs_mutated", "warning"]})
"""))

cells.append(markdown("### Стадия 2 — reward reviewers и независимые evaluators"))
cells.append(code(r"""
from mostgen.reviewers import train_reviewers

model_result = train_reviewers(config, DATA_DIR / "reviewer_training.csv", MODELS_DIR)
append_event(OUTPUT_DIR / "events.jsonl", "train-reviewers", {"rows": model_result["rows"]})
display({
    "rows": model_result["rows"],
    "evidence_tiers": model_result["evidence_tiers"],
    "independent_instances": model_result["independent_instances"],
    "reward_algorithm": model_result["reward"]["algorithm"],
    "evaluator_algorithm": model_result["evaluator"]["algorithm"],
})
"""))

cells.append(markdown("### Стадия 3 — family-specific REINVENT4/LibInvent hand-off"))
cells.append(code(r"""
from mostgen.search import write_generator_manifests

generator_result = write_generator_manifests(config, GENERATOR_DIR)
append_event(OUTPUT_DIR / "events.jsonl", "train-generator", generator_result)
display(generator_result)
"""))

cells.append(markdown("### Стадия 4 — три matched-budget стратегии генерации"))
cells.append(code(r"""
from mostgen.search import run_methods

generated_path = OUTPUT_DIR / "generated.csv"
search_result = run_methods(
    config,
    DATA_DIR / "reaction_library.csv",
    MODELS_DIR / "reward" / "reviewers.pkl",
    generated_path,
    list(config["execution"]["methods"]),
)
append_event(OUTPUT_DIR / "events.jsonl", "generate", search_result)
display(search_result)
"""))

cells.append(markdown("### Стадия 5 — метрики, reward diagnostics и абляции"))
cells.append(code(r"""
import pandas as pd
from mostgen.metrics import compute_metrics

metrics_result = compute_metrics(generated_path, DATA_DIR / "reviewer_training.csv", METRICS_DIR, config)
display(pd.read_csv(METRICS_DIR / "metrics_summary.csv"))
display(pd.read_csv(METRICS_DIR / "reward_diagnostics.csv"))
"""))

cells.append(markdown("### Стадия 6 — независимый review и физический oracle queue"))
cells.append(code(r"""
from mostgen.review import review_generated

review_result = review_generated(
    config, generated_path, MODELS_DIR / "evaluator" / "reviewers.pkl", REVIEW_DIR
)
append_event(OUTPUT_DIR / "events.jsonl", "review", {
    "reviewed": review_result["reviewed"],
    "selected": review_result["selected_after_physical_oracle"],
})
display(review_result)
"""))

cells.append(markdown("### Стадия 7 — отчёт, презентация, acceptance и checksums"))
cells.append(code(r"""
from mostgen.reporting import build_reports
from mostgen.validation import verify_experiment
from mostgen.provenance import write_artifact_manifest

reports = build_reports(OUTPUT_DIR, config)
verification = verify_experiment(OUTPUT_DIR, config)
artifact_manifest = write_artifact_manifest(OUTPUT_DIR)
display({
    "acceptance_passed": verification["passed"],
    "checks": verification["checks"],
    "artifact_count": len(artifact_manifest["artifacts"]),
    "reports": reports,
})
"""))

cells.append(markdown(r"""
## 7. Диагностика результата

Эта ячейка отвечает на вопрос «работает ли вообще» на уровне наблюдаемых
инвариантов: точный budget, уникальность, семейное покрытие, отсутствие hard
alerts, плотность reward и распределение причин отказа. Она не превращает
синтетические predictions в физические измерения.
"""))

cells.append(code(r"""
from collections import Counter

generated = read_csv(generated_path)
run_counts = Counter((row["method_id"], row["seed"]) for row in generated)
unique_counts = {
    key: len({row["smiles"] for row in generated if (row["method_id"], row["seed"]) == key})
    for key in run_counts
}
failure_counts = Counter()
for row in generated:
    for reason in filter(None, row.get("failure_reasons", "").split(";")):
        failure_counts[reason] += 1

diagnostic = {
    "rows": len(generated),
    "exact_run_counts": dict(run_counts),
    "unique_per_run": unique_counts,
    "families": dict(Counter(row["family"] for row in generated)),
    "psoralen_alerts": sum(row["psoralen_alert"].lower() == "true" for row in generated),
    "known_phototoxic_matches": sum(row["known_phototoxic_match"].lower() == "true" for row in generated),
    "nonzero_rewards": sum(float(row["reward"]) > 0 for row in generated),
    "joint_pass_reward_models": sum(row["joint_pass"].lower() == "true" for row in generated),
    "top_failure_reasons": failure_counts.most_common(),
    "independent_provisional": review_result["independent_joint_pass"],
    "final_selected": review_result["selected_after_physical_oracle"],
}
display(diagnostic)
assert verification["passed"]
"""))

cells.append(markdown(r"""
## 8. Что использовано и что обязательно улучшить

### Уже работает корректно как программный прототип

- RDKit: sanitization, canonical SMILES, формула пары, Morgan/ECFP, Tanimoto,
  Murcko scaffold, descriptors и substructure veto.
- NumPy/scikit-learn: family-aware Random Forest reward ensembles и отдельные
  Extra Trees evaluators; uncertainty как межмодельный разброс.
- Три одинаково бюджетированных поиска в одной библиотеке; deterministic seeds.
- AUC/λc/Beer–Lambert, LCB/UCB, AD, dense geometric reward, diversity penalty.
- Неизменяемые исходники, source registry, event/config/model cards и SHA-256.
- Fail-closed безопасность и отсутствие заявления о соответствии ISO.

### Критические научные ограничения текущей версии

1. **Нет реальных обучающих наблюдений.** Метки — гладкие детерминированные
   функции дескрипторов. На них можно тестировать код, но нельзя выбирать
   соединения для синтеза.
2. **CPU `libinvent_rl` — не neural LibInvent.** Это curriculum bandit над
   конечной библиотекой. Настоящий REINVENT4 запускается только после установки
   v4.8, pin/checksum reaction prior и внешнего scorer environment.
3. **Evaluator независим алгоритмически, но не по источнику данных.** Он обучен
   на тех же synthetic train records другим алгоритмом. Нужен независимый
   experimental holdout или внешний dataset.
4. **Uncertainty не откалибрована.** Разброс деревьев/ensemble не гарантирует
   coverage 90%. Нужны conformal calibration, reliability curves и интервалы по
   scaffold-held-out данным.
5. **Синтетическая правдоподобность ограничена.** Два substituent templates и
   complexity SA proxy не заменяют atom-mapped reaction validation,
   retrosynthesis (AiZynthFinder/SynthSense), availability/price и impurity risk.
6. **Safety gate неполон.** SMARTS и небольшой exact-match list полезны как
   veto, но U07–U17 reviewers сейчас не обучены на реальных endpoint datasets.
   Нужны расширенный phototoxicant registry, metabolites/photo-products,
   sensitization, irritation, Kp/Jmax и эксперимент OECD TG 432.
7. **Нет физического oracle.** XYZ-конформеры и команды подготовлены, но xTB,
   xtb4stda и stda отсутствуют. Нужны фактические GFN2-xTB энергии, broadened
   sTDA-xTB spectrum, проверка протонирования/растворителя и обработка failures.
8. **Нет доказательства в плёнке.** Раствор, агрегирование, матрица, загрузка,
   photobleaching и optical path могут полностью изменить spectrum. ISO 24444 и
   ISO 24443 относятся к готовому продукту, не к отдельной молекуле.

### Рекомендуемый порядок доработки

Сначала заменить fixtures на очищенные M01/M03/M04/M11–M13 и U07–U17 со
scaffold/time/source splits; затем откалибровать reviewers и AD; после этого
подключить pinned REINVENT4 prior и настоящие family-specific RL runs; далее
провести независимый xTB/sTDA top-50–100 review; лишь затем синтезировать малый
набор и измерить spectrum, ΔH, t½, cycling и phototoxicity. До выполнения этих
шагов корректное название результата — **исследовательский вычислительный
скрининг**, не найденный безопасный UV-фильтр.
"""))

cells.append(markdown(r"""
## 9. Выходные файлы

- `generated.csv` — все структуры, uncertainty, AD, alerts и pass/fail reasons;
- `metrics/` — matched-budget метрики, bootstrap, абляции и reward diagnostics;
- `review/cards/` — независимые карточки top-кандидатов;
- `review/physical_oracle/` — XYZ и очередь GFN2/sTDA-xTB;
- `report/report.md` и `report/presentation_7min.md`;
- `verification.json` и `artifact_manifest.json`.

Ноль финально выбранных соединений при недоступном physical oracle является
правильным fail-closed результатом, а не ошибкой генерации.
"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "venv (MOSTGen)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12", "mimetype": "text/x-python", "codemirror_mode": {"name": "ipython", "version": 3}, "pygments_lexer": "ipython3", "nbconvert_exporter": "python", "file_extension": ".py"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"wrote {NOTEBOOK} with {len(cells)} cells")
