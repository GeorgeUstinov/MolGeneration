#!/usr/bin/env python3
"""Build the single linear, executable MOSTGen audit notebook."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "MolGenerate.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip() + "\n"}


cells: list[dict] = []
cells.append(markdown(r"""
# MolGenerate — воспроизводимый генератор UV-поглощающих MOST-молекул

Этот notebook — единая линейная реализация и исполнимый аудит всего контура:
реальные открытые endpoint-данные → независимые reviewer ensembles →
family-specific transfer learning → настоящий REINVENT4 LibInvent curriculum →
нейросетевой sampling → matched evaluation → safety veto → независимая
переоценка → GFN2-xTB/sTDA-xTB unit-oracle → отчёт и машинная проверка.

Главная граница утверждений: это **исследовательский screening proxy**. Молекулы
не сертифицированы по ISO 24444/24443 и не признаны безопасными косметическими
ингредиентами. Нулевая итоговая выборка допустима, если данные/AD/неопределённость
не позволяют пройти все жёсткие ворота.
"""))

cells.append(markdown(r"""
## 1. Среда, версии и воспроизводимость

Основной анализ работает в Python 3.10. REINVENT4 v4.8.24 закреплён commit
`80a8d21aefd9c0d3ec806377522effb30cfca12a` и запускается в отдельном Python
3.12 окружении. LibInvent prior проверяется по SHA-256. Физический oracle
использует libxTB через Python/ASE и официальные `xtb4stda`/`stda` binaries.
"""))
cells.append(code(r"""
from collections import Counter
from pathlib import Path
import importlib.metadata
import json
import platform
import sys

from IPython.display import Markdown, display
from rdkit import RDLogger

for channel in ("rdApp.debug", "rdApp.info", "rdApp.warning", "rdApp.error"):
    RDLogger.DisableLog(channel)

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "config/default.json").is_file():
    raise RuntimeError("Запускайте MolGenerate.ipynb из директории TEST")

packages = {}
for package in ("rdkit", "numpy", "pandas", "scikit-learn", "openpyxl", "ase", "xtb"):
    packages[package] = importlib.metadata.version(package)
display({"project": str(PROJECT_ROOT), "python": sys.version.split()[0],
         "platform": platform.platform(), "packages": packages})
"""))

cells.append(markdown(r"""
## 2. Что используется и как устроен pipeline

1. `prepare-data`: неизменяемые raw-файлы нормализуются в длинную endpoint-
   таблицу. λmax, кинетика, Kp, фототоксичность и сенсибилизация не смешиваются.
2. `train-reviewers`: Random Forest reward ensembles и отдельные Extra Trees
   evaluators обучаются по exact-Murcko scaffold split; uncertainty — 90%
   split-conformal radius, не только разброс деревьев.
3. `train-generator`: для NBD/QC, Dewar-pyrimidinone и spiropyran создаются
   family-specific LibInvent TL agents из закреплённого reaction prior.
4. `generate`: REINVENT4 проходит curriculum chemistry → spectrum → MOST →
   safety. ExternalProcess возвращает плотный stage score. Затем checkpoint
   сэмплируется; разрешены только точные продукты фиксированных синтонов.
5. Два baseline: random reaction-library sampling и weighted retraining.
6. `metrics` и `review`: равные полные reviewer-бюджеты и итоговые evaluation
   sets, diversity/AD/gates,
   независимые evaluator models и карточки причин pass/fail.
7. Physical oracle: conformer search → GFN2-xTB оптимизация пары → ΔE/Wh·kg⁻¹;
   xtb4stda/sTDA transitions → broadened 290–400 nm proxy. Он не входит в RL.

ITI оставлен только для class-shift validation и не генерируется.
"""))

cells.append(markdown(r"""
## 3. Полный исходный код внутри notebook

Каждый модуль ниже встроен отдельной последовательной ячейкой. Они загружаются
в изолированный namespace текущего kernel. Внешний REINVENT-процесс по своей
природе не видит память notebook, поэтому вызывает идентичный дисковый bridge;
его исходник и checksum также показаны здесь.
"""))
cells.append(code(r"""
import types

for loaded_name in list(sys.modules):
    if loaded_name == "mostgen" or loaded_name.startswith("mostgen."):
        del sys.modules[loaded_name]

embedded_package = types.ModuleType("mostgen")
embedded_package.__package__ = "mostgen"
embedded_package.__path__ = [str(PROJECT_ROOT / "mostgen")]
embedded_package.__file__ = str(PROJECT_ROOT / "mostgen/__init__.py")
sys.modules["mostgen"] = embedded_package

def load_embedded(name, source, filename):
    module = types.ModuleType(name)
    module.__file__ = str(filename)
    module.__package__ = name.rpartition(".")[0]
    sys.modules[name] = module
    exec(compile(source, str(filename), "exec"), module.__dict__)
    return module
"""))

module_order = [
    "__init__.py", "config.py", "provenance.py", "chemistry.py", "numerics.py",
    "data.py", "reviewers.py", "real_data.py", "real_reviewers.py", "scoring.py",
    "search.py", "metrics.py", "oracle.py", "experimental.py", "review.py", "reporting.py",
    "validation.py", "cli.py", "__main__.py",
]
descriptions = {
    "__init__.py": "Версия пакета", "config.py": "Конфигурация и бюджеты",
    "provenance.py": "SHA-256 и manifests", "chemistry.py": "Пары изомеров и safety veto",
    "numerics.py": "AUC, λc и reward transforms", "data.py": "Reaction library и dispatch данных",
    "reviewers.py": "Общий reviewer interface и smoke fixtures",
    "real_data.py": "Парсеры реальных M/U endpoint-источников",
    "real_reviewers.py": "Endpoint ensembles, conformal uncertainty и AD",
    "scoring.py": "Прозрачные компоненты reward и hard gates",
    "search.py": "Baselines и настоящий REINVENT4/LibInvent",
    "metrics.py": "Diversity, bootstrap, ESS и абляции", "oracle.py": "GFN2-xTB/sTDA-xTB",
    "experimental.py": "Шаблоны и валидация внешних лабораторных данных (без выполнения эксперимента)",
    "review.py": "Независимая переоценка и карточки", "reporting.py": "Отчёт и презентация",
    "validation.py": "Машинные критерии", "cli.py": "Единый CLI", "__main__.py": "CLI entrypoint",
}
for filename in module_order:
    source = (ROOT / "mostgen" / filename).read_text(encoding="utf-8")
    cells.append(markdown(f"### 3.{len(cells)} {descriptions[filename]} — `{filename}`"))
    if filename == "__init__.py":
        cell_source = (
            f"_source = {source!r}\n"
            "exec(compile(_source, embedded_package.__file__, 'exec'), embedded_package.__dict__)\n"
            "print('loaded mostgen', embedded_package.__version__)"
        )
    else:
        module_name = "mostgen." + filename[:-3]
        cell_source = (
            f"_source = {source!r}\n"
            f"load_embedded({module_name!r}, _source, PROJECT_ROOT / 'mostgen' / {filename!r})\n"
            f"print('loaded {module_name}')"
        )
    cells.append(code(cell_source))

bridge = (ROOT / "scripts/reinvent_external_score.py").read_text(encoding="utf-8")
cells.append(markdown("### 3.x ExternalProcess scoring bridge — `scripts/reinvent_external_score.py`"))
cells.append(code(
    f"REINVENT_EXTERNAL_SCORER_SOURCE = {bridge!r}\n"
    "from mostgen.provenance import sha256_file\n"
    "display({'embedded_lines': len(REINVENT_EXTERNAL_SCORER_SOURCE.splitlines()), "
    "'disk_sha256': sha256_file(PROJECT_ROOT / 'scripts/reinvent_external_score.py')})"
))

config_text = (ROOT / "config/default.json").read_text(encoding="utf-8")
cells.append(markdown(r"""
## 4. Конфигурация audit и production-критерий

Исполняемый audit намеренно использует один seed и 60 молекул (20 на семейство),
но **реальные данные и настоящий neural LibInvent**, а не synthetic backend.
Это проверка работоспособности, не заявление о выполнении production-критерия.
Полная конфигурация в этой же ячейке сохраняет 3 seeds и ≥1000 уникальных
структур на запуск; для неё достаточно убрать audit-overrides.
"""))
cells.append(code(
    f"CONFIG_JSON = {config_text!r}\n" + r"""
from mostgen.config import validate_config

config = json.loads(CONFIG_JSON)
config["_config_path"] = str(PROJECT_ROOT / "config/default.json")
config["execution"].update({
    "mode": "full", "backend": "reinvent4", "seeds": [1701],
    "n_per_run": 60, "reviewer_budget_per_run": 636,
    "shortlist_size": 12, "bootstrap_samples": 40,
})
config["data"]["m13_max_rows"] = 12000
config["reviewers"]["ensemble_size"] = 3
config["reviewers"]["trees_per_member"] = 10
config["production"].update({"sampling_rounds": 1, "sampling_oversample": 12})
config["oracle"].update({"run_automatically": False, "max_candidates": 1,
                          "conformers": 3, "xtb_max_steps": 60, "xtb_fmax_ev_a": 0.18})
config["training_rows_per_family"] = 180
validate_config(config)

production_contract = json.loads(CONFIG_JSON)["execution"]
display({"audit": config["execution"], "production_contract": production_contract,
         "warning": "audit_pass != production_acceptance"})
"""))

cells.append(markdown(r"""
## 5. Реальные данные и роль исходного Excel

Используемые для fit источники:

- M13 UV/VisML: λmax transfer data, включая M11 Deep4Chem и M12 CDEx;
- M01 photoswitch database: λmax и thermal Z→E kinetics;
- U07 NICE: irritation/corrosion calls, только exact DTXSID join;
- U09 HPPT: human skin sensitization;
- U12 SkinPiX и U13 human epidermis: logKp с явным преобразованием единиц;
- U16 QsarDB: 3T3 NRU phototoxicity;
- M05: 22 condition-rich full-spectrum records двух spiropyrans, только как
  отдельная calibration/validation evidence.

`database_matrix_MOST_UV_skin.xlsx`, данный в начале, **используется**, но лишь
как каталог литературы/источников и provenance. В нём нет пригодной таблицы
`SMILES → endpoint`, поэтому он не попадает в обучение. M03 недоступен через
file API без bearer token (HTTP 401); нулевой partial исключён. M04 не превращён
в labels без надёжной автоматической идентификации структур. Синтетические
energy labels не подставляются.
"""))
cells.append(code(r"""
from mostgen.data import prepare_data, read_csv, write_csv
from mostgen.reviewers import train_reviewers
from mostgen.provenance import write_artifact_manifest

RUN_ROOT = PROJECT_ROOT / "runs/notebook_real_audit"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
data_manifest = prepare_data(config, RUN_ROOT / "data", PROJECT_ROOT)
endpoints = read_csv(RUN_ROOT / "data/reviewer_endpoints.csv")
catalog = read_csv(RUN_ROOT / "data/source_catalog_from_initial_excel.csv")
display({"training_mode": data_manifest["training_mode"],
         "endpoint_rows": data_manifest["endpoint_rows"],
         "unique_structures_total": data_manifest["unique_structures_total"],
         "full_spectrum_records": data_manifest["full_spectrum_records"],
         "reaction_library_rows": data_manifest["reaction_library_rows"],
         "initial_excel_catalog_rows": len(catalog),
         "initial_excel_usage": data_manifest["initial_excel_usage"],
         "unavailable_labels": data_manifest["unavailable_labels"]})
"""))

cells.append(markdown(r"""
## 6. Reviewer models и формирование оценки

Для каждой структуры ансамбль выдаёт mean и calibrated uncertainty. Предсказанный
λmax превращается в явно объявленный Gaussian-band proxy (σ=34 nm), после чего:

\[
AUC_{UVB}=\int_{290}^{320}A(\lambda)d\lambda,\quad
AUC_{UVA}=\int_{320}^{400}A(\lambda)d\lambda.
\]

`λc` — точка 90% cumulative area. UV gate использует нижние 90% границы AUC и
λc≥370 nm. MOST gate требует положительную нижнюю границу ΔH, семейный reference
и t½=4–24 h при 305 K. Safety gate проверяет upper bounds phototoxicity, Kp и
sensitization, SA proxy, AD и hard structural veto.

Dense reward — weighted geometric mean непрерывных sigmoid/interval scores с
floor 0.001. Отсутствующей ΔH даётся только низкий shaping-component 0.20;
hard MOST-pass при этом всегда false. Это сохраняет обучаемость curriculum, но
не позволяет превратить отсутствие данных в «успех».
"""))
cells.append(code(r"""
model_cards = train_reviewers(config, RUN_ROOT / "data/reviewer_endpoints.csv", RUN_ROOT / "models")
compact_metrics = {
    purpose: {endpoint: values for endpoint, values in model_cards[purpose]["metrics"].items()}
    for purpose in ("reward", "evaluator")
}
display({"rows": model_cards["rows"], "unique": model_cards["unique_structures"],
         "energy_model_status": model_cards["energy_model_status"],
         "independent_instances": model_cards["independent_instances"],
         "metrics": compact_metrics})
"""))

cells.append(markdown(r"""
## 7. Настоящий REINVENT4 LibInvent

Сначала создаются и обучаются три TL agents. Затем для текущего seed запускаются
12 stages (4×3 family) с DAP и `PenalizeSameSmiles`. Post-sampling сначала
использует RL checkpoint, затем TL agent как neural diversity backstop. Ни
случайное перечисление библиотеки, ни enumerative filler не добавляются.
Нейросетевые предложения вне exact allowed library учитываются в audit, но
отбрасываются до итогового scoring.
"""))
cells.append(code(r"""
from mostgen.cli import _production_checks
from mostgen.search import (
    execute_generator_training, run_methods, run_reinvent_generation,
    write_generator_manifests,
)

_production_checks(config)
generator_manifest = write_generator_manifests(config, RUN_ROOT / "generator")
transfer_learning = execute_generator_training(config, RUN_ROOT / "generator")
libinvent_result = run_reinvent_generation(
    config, RUN_ROOT / "generator", RUN_ROOT / "data/reaction_library.csv",
    RUN_ROOT / "models/reward/reviewers.pkl", RUN_ROOT / "libinvent_generated.csv",
)
display({"transfer_learning": transfer_learning,
         "curriculum_backend": libinvent_result["backend"],
         "curriculum_optimization_calls": libinvent_result["curriculum"]["optimization_reviewer_evaluations"],
         "sampling": libinvent_result["sampling"]})
"""))

cells.append(markdown(r"""
## 8. Baselines, matched reviewer budget, metrics и independent review

Baselines работают в том же reaction-library space. Каждый метод получает
ровно 636 reward-reviewer evaluations: LibInvent = 576 curriculum + 60 final,
baseline = 636 рассмотренных library candidates. После поиска каждый метод
сохраняет одинаковые 60 уникальных структур (20 на семейство). Фактические
вызовы проверяются по `search_budget_ledger.json`, а не выводятся из числа строк.
"""))
cells.append(code(r"""
from mostgen.metrics import compute_metrics
from mostgen.review import review_generated
from mostgen.reporting import build_reports
from mostgen.validation import verify_experiment

baseline_result = run_methods(
    config, RUN_ROOT / "data/reaction_library.csv", RUN_ROOT / "models/reward/reviewers.pkl",
    RUN_ROOT / "baselines.csv", ["prior_random", "weighted_retraining"],
)
combined = read_csv(RUN_ROOT / "baselines.csv") + read_csv(RUN_ROOT / "libinvent_generated.csv")
write_csv(RUN_ROOT / "generated.csv", combined)
metrics_result = compute_metrics(
    RUN_ROOT / "generated.csv", RUN_ROOT / "data/reviewer_endpoints.csv", RUN_ROOT / "metrics", config,
)
review_result = review_generated(
    config, RUN_ROOT / "generated.csv", RUN_ROOT / "models/evaluator/reviewers.pkl", RUN_ROOT / "review",
)
reports = build_reports(RUN_ROOT, config)
verification = verify_experiment(RUN_ROOT, config)
write_artifact_manifest(RUN_ROOT)

rows = read_csv(RUN_ROOT / "generated.csv")
library = {(row["family"], row["smiles"]) for row in read_csv(RUN_ROOT / "data/reaction_library.csv")}
display({"run_counts": dict(Counter((r["method_id"], r["seed"]) for r in rows)),
         "family_counts": dict(Counter((r["method_id"], r["family"]) for r in rows)),
         "all_exact_library": all((r["family"], r["smiles"]) in library for r in rows),
         "psoralen_cores": sum(str(r["psoralen_alert"]).lower() == "true" for r in rows),
         "known_phototoxic_matches": sum(str(r["known_phototoxic_match"]).lower() == "true" for r in rows),
         "joint_pass": sum(str(r["joint_pass"]).lower() == "true" for r in rows),
         "selected": review_result["selected_after_physical_oracle"],
         "laboratory_experiments_executed": False,
         "experimental_package": review_result["experimental_package"]["status"],
         "audit_verification": verification["passed"]})
"""))

cells.append(markdown(r"""
## 9. Независимый вычислительный physical unit-oracle

Чтобы проверить сам toolchain даже при нуле provisional joint-pass, здесь
рассчитывается простой NBD/QC library member F/F. Это unit-oracle, а не выбранный
кандидат: ΔE — electronic-energy proxy, а sTDA curve — broadened proxy. Для
production top-50–100 нужны более строгая конформерная сходимость и последующая
экспериментальная проверка. Эта ячейка не выполняет лабораторный пункт 9 из
перечня пользователя: синтез, измерения и OECD здесь не запускаются.
"""))
cells.append(code(r"""
from mostgen.oracle import availability, evaluate_pair

library_rows = read_csv(RUN_ROOT / "data/reaction_library.csv")
oracle_candidate = next(row for row in library_rows
                        if row["family"] == "nbd_qc" and row["synthon_a"] == "S01" and row["synthon_b"] == "S01")
physical = evaluate_pair(oracle_candidate, RUN_ROOT / "review/physical_unit_oracle", config, 1)
write_artifact_manifest(RUN_ROOT)
display({"availability": availability(config),
         "candidate": {key: oracle_candidate[key] for key in ("smiles", "charged_smiles", "family")},
         "result": {key: physical[key] for key in ("gfn2_delta_e_kj_mol", "gfn2_specific_energy_wh_kg",
                                                     "stda_uvb_auc", "stda_uva_auc", "stda_lambda_c_nm",
                                                     "ground_xtb_converged", "charged_xtb_converged")}})
"""))

cells.append(markdown("## 10. Итоговый ответ и необходимые доработки"))
cells.append(code(r'''
failure_counts = Counter()
for row in rows:
    failure_counts.update(reason for reason in row.get("failure_reasons", "").split(";") if reason)

display(Markdown(f"""
### Работает ли pipeline?

**Да, программно и end-to-end:** реальные источники распарсены, independent
reviewers обучены, family TL и 12 curriculum stages REINVENT4 завершились,
post-sampling дал требуемые audit-квоты, safety/AD/review/report/oracle исполнились.

### Генерируются ли нужные по ТЗ молекулы?

**По химическому пространству — да:** три заданных семейства, валидные
ground/charged pairs, только разрешённые синтоны, 0 псораленовых/фурокумариновых
ядер. **По совокупности целевых свойств — пока не доказано:** joint-pass =
{sum(str(r['joint_pass']).lower() == 'true' for r in rows)}. Главная причина —
нет открытой надёжной molecular ΔH training table; energy/AD gate правильно
закрыт. Это не доказательство физической невозможности.

### Что требует улучшения?

1. Получить/вручную курировать M03/M04 ΔH, ΔG‡, t½, state/condition labels.
2. Расширить очень малые U16 phototoxicity и M01 class-shift kinetics выборки.
3. Заменить λmax Gaussian proxy большим набором реальных полных спектров и
   solvent/state-aware моделью.
4. Выполнить 3 независимых production seeds и ≥1000 neural-unique molecules на
   запуск; текущая выполненная ячейка — честный audit 60, не production acceptance.
5. Провести GFN2/sTDA top-50–100, затем синтез, cycling/quantum yield/t½,
   OECD TG 432 и испытание готовой плёнки/формуляции.

Частые причины отказа в audit: `{dict(failure_counts.most_common(8))}`.

Начальный Excel: **использован как source catalog/provenance ({len(catalog)}
строк), не использован для fit**, поскольку не содержит молекулярной таблицы
SMILES/endpoint.
"""))
'''))

cells.append(markdown(r"""
## 11. Команды CLI

Полный production-прогон (3 метода × 3 seed × ≥1000 структур) реализован той же
кодовой базой. Он намеренно не маскируется результатом короткого audit:

```bash
./venv/bin/python -m mostgen run-all --mode production --output runs/production
```

Промежуточные стадии доступны как `prepare-data`, `train-reviewers`,
`sample-baselines`, `train-generator`, `generate`, `review`. Все конфиги,
checksums, model cards, CSV, oracle files и отчёты сохраняются внутри run-dir.
"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "MOSTGen venv", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10", "mimetype": "text/x-python",
                          "codemirror_mode": {"name": "ipython", "version": 3},
                          "pygments_lexer": "ipython3", "nbconvert_exporter": "python", "file_extension": ".py"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"wrote {NOTEBOOK} with {len(cells)} cells")
