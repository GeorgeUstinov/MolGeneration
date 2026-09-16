# Экспериментальная валидация MOST/UV-кандидатов

## Граница автоматизации

Код может подготовить shortlist, спецификацию образцов, пустые таблицы,
контроль единиц и проверку полноты. Он не синтезирует вещества и не выполняет
OECD TG 432. Статус `selected` до получения экспериментальных данных означает
только исследовательский приоритет, не безопасность и не соответствие ISO.

## 1. Preregistration и образцы

До начала фиксируются candidate IDs, структуры обоих состояний, отрицательные и
положительные controls, критерии исключения, число независимых синтезов и план
статистики. Для каждого кандидата нужны минимум три независимых replicate/batch.
До фотохимии подтверждаются NMR, HRMS/LC–MS, HPLC purity ≥95%, вода/остаточные
растворители и отсутствие смешения ground/charged state.

## 2. Solution UV–Vis и фотоконверсия

Для обоих состояний измеряется 250–500 nm (обязательное покрытие 290–400 nm) в
кварцевых кюветах с blank/dark correction, известной длиной пути и серией
концентраций для проверки линейности Beer–Lambert. Сохраняются исходные файлы и
SHA-256. Рассчитываются ε(λ), UVB/UVA AUC, λc, spectral overlap и PSS при 305,
365 и 395 nm. Quantum yield определяется с калиброванной actinometry, а не по
одной разности absorbance.

## 3. Кинетика хранения

Thermal back-conversion измеряется минимум при четырёх температурах вокруг
305 K в темноте. Fit выполняется для заранее выбранной кинетической модели;
сохраняются residuals и R². Arrhenius/Eyring fit даёт t½(305 K) и uncertainty.
Кандидат проходит целевое окно только если доверительный интервал совместим с
4–24 h и нет необратимых продуктов.

## 4. Энергия и тепловыделение

DSC либо валидированная solution calorimetry выполняется при известной charged
fraction. Blank, matrix и non-switching controls обязательны. ΔH переводится в
Wh·kg⁻¹ по фактической молярной массе и charged fraction; нужны минимум три
независимых replicate и uncertainty budget. Электронная GFN2 ΔE используется
лишь для ранжирования и не заменяет калориметрию.

## 5. Cycling и фотопродукты

Проводятся серии 10, 100, 500 и 1000 charge/discharge cycles при фиксированной
photon dose. После контрольных точек измеряются retained capacity, UV–Vis,
HPLC/LC–MS mass balance и новые photoproducts. Изменение baseline или появление
неидентифицированных фотопродуктов считается причиной остановки.

## 6. Плёнка и формуляция

После solution-stage готовится матрица без кожи/животных: минимум три загрузки и
три толщины, с blank matrix и reference UV filter. Измеряется hemispherical
transmittance 290–400 nm до/после charging и после cycling, толщина и
однородность. Molecular Beer–Lambert proxy сравнивается с наблюдаемой плёнкой,
но не выдаётся за ISO результат. ISO 24443 относится к готовому продукту; ISO
24444 in vivo рассматривается только после safety review и необходимых
этических/регуляторных разрешений.

## 7. Safety и OECD TG 432

Сначала выполняются solubility/stability, dark cytotoxicity и оценка достижимой
экспозиции. Затем компетентная лаборатория проводит актуальную принятую версию
3T3 NRU phototoxicity study с ±UVA dose-response, controls и quality criteria.
В таблицу заносятся IC50, PIF/MPE, classification и checksum подписанного отчёта.
При неоднозначном результате кандидат не продвигается. Дополнительно требуются
skin penetration, irritation, sensitization и оценка photoproducts.

## 8. Передача данных обратно в pipeline

`mostgen.experimental.write_experimental_package` создаёт семь таблиц.
Лаборатория не меняет названия колонок, использует указанные SI/явные единицы и
прикладывает raw-file hashes. `validate_experimental_results` проверяет schema,
identity/purity, связь candidate IDs, ≥3 replicate и спектральное покрытие.
Только валидированный пакет может стать новым versioned dataset; train/test
разделение делается по scaffold и source/batch до следующего fit.
