# Анализ multiclass photoswitch generation

## Результат проверки

В исходной реализации `Unified_Multiclass_Photoswitch_RL.ipynb` запрошенные
генеративные метрики не рассчитывались. Она сохраняла 19 626 прошедших фильтры
структур без отдельного top-1000. В общей реализации `mostgen.metrics` были
дополнительные расхождения с заданными формулами: Uniqueness делилась на число
всех сгенерированных строк, Novelty считала дубликаты, а Diversity оценивалась
по выборке максимум из 500 пар.

Теперь определения унифицированы:

- `Validity = N_valid / N_generated`;
- `Uniqueness = N_unique_valid / N_valid`;
- `Novelty = N_unique_valid_not_in_training / N_unique_valid`;
- `JSR = N(target_A_pass AND target_B_pass) / N_generated` (невалидные строки
  считаются неуспехом);
- `Diversity = 1 - 2 / (N(N-1)) * sum_{i<j} Tanimoto(x_i, x_j)` по всем парам
  уникальных валидных Morgan fingerprint (radius 2, 2048 bit), без sampling.

Novelty сравнивается с 1 153 уникальными структурами, фактически вошедшими в
SELFIES training corpus, а не с validation/test.

## Метрики финального запуска

| Scope | Family | Ngenerated | Nvalid | Nunique | Nnot in training | Njoint | Validity | Uniqueness | Novelty | JSR | Diversity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all proposals | all | 36 000 | 35 998 | 26 568 | 26 564 | 19 942 | 0.999944 | 0.738041 | 0.999849 | 0.553944 | 0.785712 |
| all proposals | azo | 6 000 | 5 999 | 5 507 | 5 505 | 1 148 | 0.999833 | 0.917986 | 0.999637 | 0.191333 | 0.808092 |
| all proposals | stilbene | 30 000 | 29 999 | 21 120 | 21 118 | 18 794 | 0.999967 | 0.704023 | 0.999905 | 0.626467 | 0.758991 |
| top 1000 | all | 1 000 | 1 000 | 1 000 | 1 000 | 1 000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.729000 |
| top 1000 | azo | 500 | 500 | 500 | 500 | 500 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.735075 |
| top 1000 | stilbene | 500 | 500 | 500 | 500 | 500 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.604556 |

Метрики `all proposals` характеризуют сам generator. Единичные метрики top-1000
являются следствием целевого отбора и не должны интерпретироваться как
несмещённая оценка генератора.

## Отбор 1000 кандидатов

Итоговый shortlist содержит ровно 500 azo и 500 stilbene, все canonical SMILES
уникальны глобально. Внутри каждого класса сортировка выполняется сначала по
совместному прохождению целей, затем по evaluator/proxy score, reward и
детерминированному canonical-SMILES tie-break. Межклассовый score напрямую не
сравнивается, потому что доступные endpoint-наборы различаются.

- Target A для обоих классов: held-out UV evaluator, `lambda_max` в диапазоне
  300–400 nm.
- Target B для azo: held-out half-life evaluator в диапазоне 4–24 h и
  M01 transition-proxy `delta-lambda >= 20 nm`.
- Target B для stilbene: валидная структурная E/Z-пара. Для stilbene отсутствуют
  half-life, PSS, quantum yield, deltaH и stored-energy labels.

Поэтому stilbene JSR — структурно-спектральный proxy и не эквивалентен по силе
доказательства azo JSR. Результаты являются computational screening candidates,
а не экспериментально подтверждёнными фотопереключателями.

## Артефакты и проверки

- `generated_multiclass_photoswitch.csv` — новый balanced top-1000;
- `candidate_pool.csv` — полный отфильтрованный пул из 20 480 кандидатов;
- `proposal_audit.csv` — все 36 000 сырых предложений и target flags;
- `generation_metrics.csv` — числители, знаменатели и метрики по scope/family;
- `run_manifest.json` — определения targets, метрик и правила отбора.

SHA-256 итогового CSV:
`c1f4dabb0858ed3af9c59b5bc0ecce70f824a82f68529b5a31eeafc2c458e3ad`.

Проверки: 46 тестов прошли; notebook завершён без error outputs; подтверждены
размер, глобальная уникальность, баланс 500/500, target flags, ранги и отсутствие
пересечений top-1000 с training corpus.
