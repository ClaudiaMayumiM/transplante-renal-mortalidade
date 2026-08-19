# Escopo de reprodutibilidade e interpretação

O conteúdo preserva publicamente o estado científico congelado. A preparação para publicação incluiu ajustes editoriais e de interoperabilidade que não modificaram fórmulas, populações analíticas, modelos, previsões ou resultados científicos. Os ajustes de interoperabilidade apenas conectam etapas já definidas do pipeline; nenhum modelo foi reajustado e nenhuma nova previsão OOF foi gerada.

O repositório, isoladamente, não reproduz integralmente as análises porque não redistribui as entradas individuais. Também não constitui validação externa, garantia de transportabilidade ou demonstração de utilidade clínica.

As conclusões são associativas e prognósticas, restritas à coorte analisada. Não há alegação causal. Categoria etária de maior idade e DGF são os sinais prognósticos centrais. A ridge pode ser descrita como numericamente mais favorável em avaliações internas específicas, nunca como superior. HR, OR e coeficientes ridge não compartilham escala numérica.

O script de classificação recebe previsões OOF previamente geradas, aplica os limiares definidos e produz contagens e métricas agregadas. Sua validação utilizou previsões históricas já congeladas: nenhum modelo foi reajustado, nenhuma previsão OOF foi regenerada e nenhum novo procedimento de reamostragem aleatória foi executado.

Alguns modos auxiliares de verificação de `run_bootstrap.py`, como `--preflight-only` e `--gate-only`, dependem de artefatos internos de auditoria não distribuídos publicamente e não são necessários para a execução das análises científicas principais.

`kaplan_meier.py` contém uma etapa intermediária de preparação e resumo. A apresentação final da figura de Kaplan-Meier é governada pelo respectivo gerador de figura e pelos artefatos agregados usados no reporting.

O horizonte binário de cinco anos foi usado apenas para avaliar a viabilidade amostral e não foi empregado na modelagem binária. Uma derivação histórica auxiliar usa aniversário calendárico; na coorte analisada, essa representação não produziu divergência de classificação em relação ao horizonte fixo adotado na análise final.

As previsões individuais usadas na validação não são distribuídas. Somente `classification_metrics_summary.csv`, agregado por análise, modelo, limiar e métrica, integra o repositório público.
