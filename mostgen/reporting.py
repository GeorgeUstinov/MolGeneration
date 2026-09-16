from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data import read_csv


def _pct(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def build_reports(experiment_dir: str | Path, config: dict[str, Any]) -> dict[str, str]:
    root = Path(experiment_dir)
    output = root / "report"
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = root / "metrics" / "metrics_summary.csv"
    metrics = read_csv(metrics_path) if metrics_path.exists() else []
    generated = read_csv(root / "generated.csv")
    review = json.loads((root / "review" / "review_summary.json").read_text(encoding="utf-8"))
    table = ["| Метод | Валидность | Уникальность | Diversity | Joint success | Обе AD |", "|---|---:|---:|---:|---:|---:|"]
    for row in metrics:
        table.append(
            f"| {row['method_id']} | {_pct(row['validity_mean'])} | {_pct(row['uniqueness_mean'])} | "
            f"{float(row['internal_diversity_mean']):.3f} | {_pct(row['joint_success_mean'])} | {_pct(row['both_ad_fraction_mean'])} |"
        )
    family_counts = {family: sum(row["family"] == family for row in generated) for family in config["families"]}
    no_joint = sum(str(row.get("joint_pass", "")).lower() == "true" for row in generated) == 0
    real_mode = bool(generated) and generated[0].get("reviewer_data_mode") == "real_endpoint_specific"
    data_manifest_path = root / "data" / "data_manifest.json"
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8")) if data_manifest_path.exists() else {}
    generator_description = (
        "настоящий REINVENT4/LibInvent staged-learning с family-specific TL agents и post-sampling"
        if real_mode else "детерминированный reaction-library smoke-surrogate"
    )
    evidence_description = (
        "Обучение выполнено на реальных endpoint-записях M13 (включая M11 Deep4Chem и M12 CDEx), M01, U07, U09, U12, U13 и U16; M05 используется как малая full-spectrum проверка. "
        "Начальный Excel используется только как каталог источников/provenance. Таблица молекулярных ΔH недоступна, поэтому energy-модель не обучена и MOST/joint gate закрыт до GFN2-xTB/эксперимента."
        if real_mode else
        "Это smoke-проверка на метках `synthetic_smoke_only`; она тестирует программную механику, но не является научным обучением."
    )
    report = f"""# Воспроизводимый скрининг UV-поглощающих MOST-фотопереключателей

## 1. Цель, область утверждений и вычислительный бюджет

Цель прототипа — найти вычислительное пересечение широкополосного поглощения 290–400 нм, молекулярного накопления энергии, времени хранения 4–24 ч при 305 K и консервативного safety-triage. Проектируется одна малая фотопереключаемая молекула, не смесь и не готовая солнцезащитная формуляция. Результаты имеют статус **screening proxy**. Они не подтверждают безопасность, эффективность, фотостабильность или пригодность вещества как косметического ингредиента.

ISO 24444:2019 описывает in vivo определение SPF готового продукта, а ISO 24443:2021 — in vitro оценку UVA-защиты продукта. Поэтому ни одна молекула здесь не называется соответствующей ISO. Следующий уровень доказательности требует изготовления плёнки/формуляции и испытания продукта. Фототоксичность требует отдельной экспериментальной проверки, например OECD TG 432.

Запуск: режим `{config['execution']['mode']}`, backend `{config['execution']['backend']}`, seeds `{config['execution']['seeds']}`. Лимиты: не более {config['project']['max_training_structures']} обучающих структур и {config['project']['max_gpu_hours']:.1f} GPU·ч; фактическая конфигурация заявляет {config['execution']['gpu_hours']:.1f} GPU·ч. Для matched comparison каждый метод получает ровно {config['execution']['reviewer_budget_per_run']} reward-reviewer evaluations на seed и сохраняет одинаковые {config['execution']['n_per_run']} структур. Для LibInvent бюджет состоит из curriculum optimization и финального scoring; фактические вызовы всех методов записаны в `search_budget_ledger.json`.

## 2. Генеративное пространство и сравниваемые методы

Три семейства запускались как family-aware пространства с фиксированными реакционными каркасами и двумя разрешёнными позициями замещения: NBD/QC как основной физически определённый класс, Dewar-pyrimidinone как малодатовый экспериментальный класс и spiropyran/merocyanine как проверка переносимости. Заряженный изомер строился детерминированно; RDKit проверял валентность, санитаризацию и равенство молекулярной формулы пары. Размеры фактически оценённых подмножеств: `{family_counts}`.

Сравнены (i) случайный reaction-constrained prior, (ii) adaptive weighted-retraining по той же библиотеке синтонов и (iii) {generator_description}. В реальном режиме neural output принимается только при точном совпадении с версионированной библиотекой допустимых продуктов; enumerative filler не используется.

Curriculum последовательно открывает валидность/семейство/пару, спектр, MOST, затем безопасность/SA/AD. Награда — взвешенное геометрическое среднее непрерывных sigmoid/interval-компонентов с floor `{config['reward']['floor']}`. ECFP-centroid filter штрафует повторное заселение кластеров. Таблица `reward_diagnostics.csv` позволяет проверить долю ненулевых наград, effective sample size, баланс компонентов и scaffold collapse.

## 3. Reviewer-модели, неопределённость и применимость

Pipeline сохраняет exact-Murcko scaffold split до обучения и проверяет отсутствие пересечения scaffold между train/validation/test. Reward-модели и независимые evaluator-модели сериализованы раздельно: первые используют ensemble Random Forest и направляют поиск, вторые — отдельный ensemble Extra Trees и переоценивают только top-кандидатов. В real mode λmax превращается в явно маркированный Gaussian-band proxy; он не выдаётся за измеренный полный спектр. Из proxy интегрируются UVB AUC, UVA AUC, critical-wavelength proxy и условная Beer–Lambert transmittance.

Joint UV-pass требует, чтобы нижняя 90%-я доверительная граница обоих AUC была не хуже медианы, нижняя граница λc-proxy была не ниже {config['reviewers']['lambda_c_min_nm']:.0f} нм, а worst-side conditional Beer–Lambert transmittance не превышала заданных UVB/UVA порогов. MOST-pass требует положительную нижнюю границу ΔH, результат не хуже семейной медианы, specific-energy LCB ≥{config['reviewers']['specific_energy_min_wh_kg']:.0f} Wh·kg⁻¹ и средний t½ в окне {config['reviewers']['half_life_window_hours'][0]:.0f}–{config['reviewers']['half_life_window_hours'][1]:.0f} ч. Высокий прогноз вне fingerprint-domain не засчитывается.

{evidence_description}

Число endpoint-записей: `{data_manifest.get('endpoint_rows', 'см. data_manifest.json')}`. M03 не был использован: API Dryad возвращает HTTP 401 без bearer token; нулевой partial-файл исключён. M04 сохранён как физический reference, но не превращён в labels без надёжной идентификации структур.

## 4. Safety gate и правила shortlist

До вычисления награды применяются SMARTS-поиск линейных и угловых фурокумариновых/псораленовых ядер, точное совпадение с локальным списком известных фототоксичных структур, запрет неподдерживаемых элементов, reactive/unstable alerts, проверка валентности и построения изомерной пары. Псоралены и иные фурокумарины не могут пройти в итоговый CSV. Это консервативно согласуется с заключением SCCP: безопасность фурокумаринов не была подтверждена, а фототоксичность нельзя считать исключённой.

Вероятность фототоксичности и Kp — лишь triage. Интервал неопределённости фототоксичности блокирует продвижение независимо от среднего значения. Отсутствие алерта не является доказательством безопасности. В карточке каждого top-кандидата раздельно показаны spectrum/MOST/safety, uncertainty, обе AD, SA-proxy, причины отказа и уровень доказательности.

## 5. Результаты и matched-budget comparison

{chr(10).join(table)}

Сгенерировано {len(generated)} строк; независимый evaluator пересмотрел {review['reviewed']}, provisional joint-pass получили {review['independent_joint_pass']}. После физического oracle выбрано {review['selected_after_physical_oracle']}. Статус oracle: `{review['physical_oracle']['status']}`. При отсутствии xTB/sTDA pipeline не подставляет суррогатный «квантовый» результат и оставляет `selected=false` — это намеренное fail-closed поведение.

В production-конфигурации bootstrap-интервалы агрегируют три seed-запуска; короткий audit с одним seed не оценивает межзапусковую устойчивость. `ablations.csv` показывает эффекты снятия uncertainty, safety, AD и diversity gates; сравнение curriculum с prior следует читать как matched-set методическое сравнение, а не как доказательство превосходства на реальных данных.

## 6. Вывод, ограничения и следующий эксперимент

{'В этом запуске не найдено кандидатов, прошедших исходный joint gate. Это допустимый научный результат: таблицы причин отказа показывают конфликт требований и неопределённость; он не доказывает физическую невозможность.' if no_joint else 'Некоторые структуры прошли внутренний reward gate, но они остаются вычислительными гипотезами и требуют независимого физического и экспериментального подтверждения.'}

Следующий этап: получить надёжно идентифицированные молекулярные ΔH/ΔG‡/t½ записи M03/M04 и дополнительные U07–U17 endpoint-данные; затем выполнить GFN2-xTB conformer review и sTDA-xTB broadened spectrum для 50–100 top-кандидатов. После синтеза нужны проверка идентичности обоих изомеров, циклируемость, quantum yield, растворитель/плёнка, OECD TG 432 и тестирование готовой формуляции по применимым стандартам.

## Приложение A. Протокол воспроизводимости и аудит данных

Единицей эксперимента считается тройка `method_id × seed × resolved configuration`. Для неё фиксируются версия Python, путь интерпретатора, версии RDKit/NumPy/Pandas/scikit-learn, платформа, команда запуска, абсолютные пути исходных файлов, SHA-256 каждого локального входа и полный registry внешних источников. URL без локально зафиксированного артефакта получают явный checksum-статус `NOT_FETCHED_NO_LOCAL_ARTIFACT`, а не фиктивный хеш. После завершения создаётся `artifact_manifest.json` с размером и SHA-256 каждого результата. Это позволяет отличить научно значимое изменение конфигурации от случайной подмены входного файла.

Исходные XLSX/DOCX/PPTX не копируются поверх и не редактируются. В real mode производная таблица `reviewer_endpoints.csv` содержит source_id, endpoint, исходные условия, состояние и label-level; `full_spectra.csv` хранится отдельно. Начальный `database_matrix_MOST_UV_skin.xlsx` распарсен в `source_catalog_from_initial_excel.csv`, но не является таблицей SMILES/labels и не входит в fit. `reaction_library.csv` — отдельное дискретное пространство генерации.

Scaffold split рассчитывается до fit. Один и тот же generic Murcko scaffold не может появиться в разных split, а loader повторно проверяет пересечение и останавливает обучение при leakage. Validation служит выбору/калибровке, test — финальной диагностике. ITI обозначен только как validation-only class shift и не включён в три генеративных семейства. Малые M03/M04 должны использоваться прежде всего для физической проверки и оценки переноса; литературные M08–M10 не трактуются как доступные виртуальные библиотеки. Регуляторные U01/U03/U06 остаются справочниками, а не автоматически размеченными обучающими строками.

## Приложение B. Как читать метрики и отрицательный результат

Validity отвечает только за машинно проверяемую химическую корректность и не равна устойчивости соединения. Uniqueness считается по каноническому SMILES внутри запуска. Novelty сравнивает результат с фактическим training set, но не с мировой химической литературой. Internal diversity — среднее расстояние Morgan fingerprints на детерминированной выборке пар; scaffold diversity — доля generic Murcko scaffold. SA-score в этом прототипе — прозрачный complexity proxy, не оценка маршрута синтеза. Поэтому высокая novelty или низкий SA не означают коммерческую доступность продукта реакции.

Joint success — наиболее строгая метрика: она требует одновременно UV-pass, MOST-pass, safety-pass и обе AD. Доверительные границы здесь важнее средних значений. Например, молекула с высоким средним UVA AUC отвергается, если ensemble расходится настолько, что нижняя граница не достигает семейного референса. Аналогично положительное среднее ΔH не помогает за пределами family-specific MOST domain. В phototoxicity применяется ещё более консервативное правило: неопределённая область сама по себе блокирует shortlist. Такой порядок уменьшает число эффектных, но неподтверждаемых виртуальных лидов.

Bootstrap выполняется по трём независимым seed-результатам, а не по 3000 молекулам как будто они независимые эксперименты. Поэтому интервал отражает вариабельность запуска метода, хотя при трёх seeds он остаётся ориентировочным. `metrics_runs.csv` сохраняет исходные значения, а `metrics_summary.csv` — среднее и percentile interval. Для сравнения методов нужно одновременно смотреть joint success и diversity: метод с немного большим success, но с collapse к одному кластеру, не обязательно предпочтительнее. `reward_diagnostics.csv` показывает effective sample size и максимальную долю одного ECFP-кластера; крайне малый ESS сигнализирует, что несколько молекул доминируют в обучающем сигнале.

Абляция uncertainty показывает, сколько кандидатов появилось бы при использовании только средних прогнозов. Абляция safety или AD не является альтернативным допустимым shortlist — это диагностическая карта конфликта требований. Абляция diversity сравнивает кластерное покрытие top-100 при ранжировании исходной и штрафованной наградой. Curriculum оценивается относительно двух matched-budget baseline, но причинный вывод о пользе curriculum требует повторения на реальных моделях и данных. Если joint-кандидатов нет, следует изучить распределения отдельных gate и failure reasons, а не ослаблять пороги постфактум.

## Приложение C. Переход от прототипа к физическому исследованию

`train-generator` создаёт отдельный REINVENT4 v{config['production']['reinvent4_version']} TOML для каждого `family × seed`, scaffold-файл с двумя attachment points и ExternalProcess bridge к reward-reviewers. Конфигурация использует family TL, staged learning, DAP, четыре curriculum stages и PenalizeSameSmiles. Production preflight требует установленный `reinvent`, существующий LibInvent prior с совпадающим SHA-256 и полный xTB/sTDA toolchain. Отсутствие обязательного артефакта завершает запуск ошибкой до выдачи научного результата.

Для physical oracle RDKit создаёт восемь конформеров обоих состояний, выполняет MMFF/UFF pre-relaxation и сохраняет лучший XYZ. Затем GFN2-xTB должен независимо оптимизировать обе структуры; разность электронных энергий не подменяет свободную энергию, но служит внешней проверкой знака и масштаба storage proxy. sTDA-xTB даёт переходы, которые необходимо уширить с явно записанной шириной линии и повторно интегрировать по UVB/UVA. Расчёты не возвращаются во внутренний RL loop, поэтому сохраняется независимость внешней проверки.

Даже успешный physical oracle не делает молекулу безопасным ингредиентом. До формуляции нужны синтез и аналитическое подтверждение структуры, выделение обоих состояний, измерение спектра и quantum yield, проверка обратимости/усталости, t½ при нескольких температурах и оценка побочных фотопродуктов. Safety-пакет должен включать растворимость, проникновение через кожу, раздражение, сенсибилизацию и экспериментальную фототоксичность. Только после этого имеет смысл изготовить воспроизводимую плёнку с контролируемой загрузкой и сравнивать готовый продукт применимыми методами ISO. Молекулярный расчёт остаётся способом приоритизации, а не заменой этой цепочки доказательств.

Источники: [REINVENT4](https://github.com/MolecularAI/REINVENT4), [SCCP opinion](https://ec.europa.eu/health/ph_risk/committees/04_sccp/docs/sccp_o_036.pdf), [ISO 24444](https://www.iso.org/standard/72250.html), [ISO 24443](https://www.iso.org/standard/75059.html), [OECD TG 432](https://www.oecd.org/en/publications/2019/06/test-no-432-in-vitro-3t3-nru-phototoxicity-test_g1gh4b69.html).
"""
    report_path = output / "report.md"
    report_path.write_text(report, encoding="utf-8")
    slides = f"""# 7-минутная презентация

## Слайд 1 — Задача и честная граница утверждений (0:00–0:45)

- Одна UV-поглощающая MOST-молекула, не готовая формуляция.
- Пересечение UVB/UVA, энергии, 4–24 ч и safety triage.
- Только screening proxy; не ISO-сертификация и не доказательство безопасности.

## Слайд 2 — Три family-aware пространства (0:45–1:35)

- NBD/QC: основной класс; Dewar-pyrimidinone: low-data; spiropyran: transferability.
- Reaction-constrained синтоны и фиксированные позиции.
- RDKit: валентность, формула изомерной пары, canonical identity.

## Слайд 3 — Matched-budget дизайн (1:35–2:25)

- Prior random vs weighted retraining vs {generator_description}.
- {config['execution']['reviewer_budget_per_run']} reward-reviewer evaluations и {config['execution']['n_per_run']} сохранённых кандидатов на method × seed.
- REINVENT4 manifests отделены от CPU smoke backend.

## Слайд 4 — Reward и reviewer independence (2:25–3:25)

- Spectrum 290–400 нм → UVB/UVA AUC, λc, Beer–Lambert proxy.
- λmax/kinetics/Kp/phototoxicity/sensitization models; ΔH intentionally missing.
- Random Forest reward ≠ Extra Trees evaluator; scaffold split + AD + uncertainty.

## Слайд 5 — Жёсткая безопасность (3:25–4:20)

- Veto псораленов/фурокумаринов до reward.
- Known phototoxic/reactive/unsupported/invalid pair — немедленный отказ.
- Uncertain phototoxicity никогда не проходит в shortlist.

## Слайд 6 — Результаты (4:20–5:40)

{chr(10).join(table)}

- Generated: {len(generated)}; independent reviewed: {review['reviewed']}.
- Provisional: {review['independent_joint_pass']}; final after oracle: {review['selected_after_physical_oracle']}.
- Oracle status: `{review['physical_oracle']['status']}`.

## Слайд 7 — Решение и следующий шаг (5:40–7:00)

- Отсутствие joint-pass — информативный результат, не доказательство невозможности.
- Добавить идентифицированные ΔH/ΔG‡ labels и расширить малые safety/kinetics endpoints.
- GFN2/sTDA-xTB top-50–100 → синтез → OECD TG 432 → испытание плёнки/формуляции.
"""
    slides_path = output / "presentation_7min.md"
    slides_path.write_text(slides, encoding="utf-8")
    return {"report": str(report_path.resolve()), "presentation": str(slides_path.resolve())}
