# Mortalidade após transplante renal: códigos das análises do trabalho

Este repositório reúne os códigos científicos associados ao trabalho sobre sinais prognósticos de mortalidade após transplante renal em uma coorte retrospectiva observacional. O modelo de riscos proporcionais de Cox é o eixo principal; representações binárias em horizontes fixos são análises complementares.

O trabalho tem natureza associativa e prognóstica. A avaliação preditiva é interna e exploratória. O material não sustenta inferência causal, validação externa, transportabilidade, finalidade clínica ou recomendação assistencial. Resultados numericamente mais favoráveis da regressão ridge em avaliações internas não demonstram superioridade. Hazard ratios, odds ratios e coeficientes penalizados pertencem a escalas diferentes e não são numericamente equivalentes.

## Estrutura

- `src/`: preparação, sobrevivência, desfechos binários, modelos, validação interna, bootstrap e reporting;
- `config/`: constantes científicas;
- `docs/`: fluxo, disponibilidade de dados, definições, proveniência e limites de reprodutibilidade;
- `environment/`: versões computacionais documentadas no trabalho e dependências de reporting identificadas;
- `outputs/reference/metrics/`: fontes agregadas, não identificáveis, incluindo o resumo público das métricas de classificação dependentes de limiar.

## Fluxo de análise

1. preparação da base analítica e preservação das categorizações definidas no estudo;
2. derivação dos desfechos e aplicação das regras de observabilidade;
3. Kaplan-Meier e modelos de Cox principal, administrativo e landmark no dia 7;
4. regressão logística, regressão ridge e árvore rasa no horizonte definido;
5. validação interna por OOF estratificado repetido;
6. classificação dependente de limiar a partir das previsões OOF já existentes;
7. bootstrap para correção de otimismo e estabilidade;
8. reporting a partir dos resultados consolidados.

A classificação usa os limiares 0,5 e prevalência de eventos no fold de treinamento que originou cada previsão. Sensibilidade, especificidade, VPP, VPN, acurácia, acurácia balanceada e F1 são calculadas por repetição OOF completa. A análise é exploratória: não há otimização de limiar nem significado clínico validado.

A ordem e o papel de cada arquivo estão descritos em `docs/analysis_workflow.md`. `threshold_classification_metrics.py` implementa a etapa de classificação dependente de limiar a partir de previsões OOF previamente geradas. A rotina foi validada contra resultados agregados congelados, sem reajuste de modelos ou regeneração de previsões OOF.

## Dados

Os dados individuais não são redistribuídos. Não há neste repositório dados linha a linha, identificadores, datas clínicas individuais ou previsões individuais. A estrutura esperada é documentada apenas no nível necessário em `docs/data_availability.md` e `docs/variable_definitions.md`.

## Ambiente computacional

O trabalho registra Python 3.12.7, NumPy 1.26.4, pandas 2.2.2, scikit-learn 1.8.0, statsmodels 0.14.2, R 4.3.1 e `survival` 3.5.5. Consulte `environment/README.md`.

## Escopo de reprodutibilidade

O repositório preserva e documenta os códigos científicos associados às análises apresentadas no trabalho. A reprodução integral depende da disponibilidade das entradas analíticas correspondentes, que não são redistribuídas.

Não execute os scripts sem autorização para usar os dados e sem revisar os caminhos de saída: algumas rotinas geram artefatos individuais localmente, os quais são ignorados pelo Git e não devem ser publicados.

## Referência ao trabalho e versionamento

- Repositório: `<URL>`
- Commit: `<SHA completo>`
- Tag: `<tag associada ao trabalho>`
