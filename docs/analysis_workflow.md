# Fluxo das análises

Este documento localiza o código congelado; não prescreve uma nova execução.

## 1. Preparação e desfechos

- `src/00_data_preparation/prepare_analytic_data.py`: prepara a base analítica, preserva a categoria etária original, deriva `dgf_main` de `dial_1stweek` e constrói os desfechos de sobrevivência e a verificação de viabilidade binária em cinco anos.
- `src/02_binary_outcomes/derive_binary_outcomes.py`: aplica a regra canônica de observabilidade nos horizontes de um e dois anos. Óbito até o horizonte é evento; vida confirmada no horizonte é não evento; censura precoce sem óbito produz outcome não observável.
- `src/01_survival/derive_landmark_data.py`: deriva as populações landmark do dia 7. No Cox landmark, o tempo reinicia depois do marco. No binário landmark de dois anos, o horizonte permanece no dia 730 contado desde o transplante.

A derivação de cinco anos serve apenas à viabilidade. Não existe modelo binário final de cinco anos.

## 2. Sobrevivência

- `src/01_survival/kaplan_meier.py`: estimativas globais e estratificadas por categoria etária e DGF, sem teste de log-rank;
- `src/01_survival/cox_primary.R`: Cox principal com categoria etária de maior idade e DGF;
- `src/01_survival/cox_administrative.R`: sensibilidade com censura administrativa em 30 de junho de 2015;
- `src/01_survival/cox_landmark.R`: sensibilidade landmark no dia 7;
- os mesmos scripts de Cox calculam os diagnósticos de riscos proporcionais utilizados.

## 3. Modelos binários e validação interna

- `src/03_binary_models/run_binary_models.py`: regressão logística sem penalização, ridge L2 com `C=1`, `lbfgs` e `max_iter=5000`, e árvore rasa exclusivamente em dois anos. O arquivo `oof_predictions.csv` produzido localmente registra, para cada previsão, a prevalência do evento no respectivo fold de treinamento;
- `src/04_internal_validation/oof_aggregation_utils.py`: métricas por repetição OOF completa e seus resumos;
- `src/04_internal_validation/model_fit_diagnostics_utils.py`: diagnóstico estático/operacional dos ajustes.

A validação usa três folds estratificados, 100 repetições e semente 42. Cada repetição completa é uma unidade de avaliação; não se empilham previsões repetidas em uma amostra pooled.

## 4. Classificação dependente de limiar

- `src/04_internal_validation/threshold_classification_metrics.py`: recebe o `oof_predictions.csv` previamente gerado, usa a prevalência registrada para o fold de treinamento que originou cada previsão e calcula, por repetição completa, matrizes de confusão para o limiar 0,5 e para essa prevalência.

São resumidas sensibilidade, especificidade, VPP, VPN, acurácia, acurácia balanceada e F1 por mediana e percentis empíricos 2,5 e 97,5, com preservação de valores indefinidos. Não há ajuste de modelos, geração de folds, regeneração OOF, otimização de threshold ou pooling entre repetições. O resumo agregado público está em `outputs/reference/metrics/classification_metrics_summary.csv`; os inputs e resultados por repetição permanecem privados.

## 5. Bootstrap

- `src/05_bootstrap/run_bootstrap.py`: motor canônico para correção de otimismo e estabilidade interna, com 1.000 tentativas não estratificadas, reamostragem por participante, reposição e semente mestre 42.

O Cox administrativo não recebe bootstrap. A árvore aparece apenas no cenário final de dois anos.

## 6. Reporting

- `src/06_reporting/generate_figure8_km.R`: Figura 8 a partir das seis fontes agregadas `km_*_APPROVED.csv`;
- `src/06_reporting/generate_figure9_calibration.R`: Figura 9 a partir da fonte agregada de calibração e da fonte da Tabela 3;
- `src/06_reporting/generate_table3_canonical.py`: Tabela 3 a partir de `table3_reporting_source.csv`.

As fontes agregadas estão em `outputs/reference/metrics/`. Elas não contêm identificadores nem previsões individuais.
