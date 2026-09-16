# Метод и научный контракт MolGenerate

## Данные

`full/production` формируют `reviewer_endpoints.csv` из фактически локальных
M13 (λmax; включает M11 Deep4Chem и M12 CDEx), M01 (photoswitch λmax и thermal
kinetics), U07 (irritation/corrosion exact-DTXSID join), U09 (human patch-test
sensitization), U12 SkinPiX и U13 human epidermis (logKp), U16 QsarDB (3T3 NRU).
M05 full spectra двух spiropyrans сохраняются отдельно как малая проверка.
Каждая строка имеет endpoint, molecule/state/condition semantics, source и
exact-Murcko split. Исходный Excel превращается только в source catalog.

M03 file API требует bearer token; нулевой partial исключён. M04 сохранён, но
не превращён в labels без надёжной связи спектров и структур.
U08/U10/U11/U14/U17 пока не дают нормализованной доступной таблицы
`SMILES → endpoint`. Литературные M08–M10 не объявляются bulk datasets.

## Reviewers

Reward = ensembles Random Forest, evaluator = независимые Extra Trees.
Validation residuals калибруют 90% split-conformal uncertainty; test scaffold
не участвует в fit/calibration. λmax переводится в объявленный Gaussian-band
proxy σ=34 nm. Kinetics M01 является class-shift evidence для MOST families.
Из-за отсутствия molecular ΔH labels energy и MOST AD не предсказываются.

## Generator

REINVENT4 v4.8.24 закреплён commit
`80a8d21aefd9c0d3ec806377522effb30cfca12a`. Официальный `libinvent.prior`
имеет SHA-256
`03e6cbe8a53e59a4ac3aa6728d041f1957bdd07b5eefdf2cfc5c8591036075af`.
Для каждого family создаётся TL agent, для каждого family×seed — staged-learning
TOML. Curriculum: chemistry → spectrum → MOST → safety; DAP и
`PenalizeSameSmiles`. ExternalProcess принимает score только для exact product
из versioned synthon library. После RL checkpoint сэмплируется нейросеть; TL
agent служит neural diversity backstop. Enumerative filler запрещён.

Сравнение использует равный полный reward-reviewer budget. В production это
1576 evaluations на method×seed: для LibInvent 576 curriculum calls + 1000
финальных scoring calls; baselines оценивают 1576 уникальных library candidates
и сохраняют те же 1000 family-balanced результатов. Audit использует
соответственно 636 = 576 + 60. Фактические числа проверяются по
`search_budget_ledger.json`.

## Score и pass

UVB/UVA AUC, λc и worst-side Beer–Lambert transmittance вычисляются из proxy.
UV-pass требует обе AUC LCB не хуже reference, λc LCB ≥370 nm и conditional
film-proxy UCB transmittance не выше конфигурируемых порогов. MOST-pass требует
положительную ΔH LCB, family reference, specific-energy LCB ≥50 Wh/kg и t½
4–24 h. Safety учитывает phototoxicity/Kp/sensitization upper bounds, SA, AD и
hard alerts. Dense reward — weighted geometric mean sigmoid/interval components
с floor; missing energy получает только shaping score 0.20 и никогда не проходит
hard gate.

## Physical oracle и эксперимент

RDKit ETKDG retry/random-coordinates + MMFF/UFF pre-relaxation предшествуют
GFN2-xTB оптимизации обоих состояний. ΔE и Wh/kg — внешние electronic proxies.
xtb4stda/sTDA transitions уширяются на 290–400 nm. Oracle не входит в RL.

Физический лабораторный пункт не выполняется кодом. Программа лишь создаёт
пустые схемы и валидирует возвращённые лабораторией identity, solution spectra,
photokinetics, calorimetry, cycling, film spectra и OECD TG 432 data. См.
`docs/EXPERIMENTAL_VALIDATION.md`.
