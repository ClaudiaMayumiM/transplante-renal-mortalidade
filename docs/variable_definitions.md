# Definições canônicas essenciais

## Categoria etária

`age_gt40_main` preserva a categorização histórica da base. Ela não deve ser reconstruída a partir da idade contínua. A categoria historicamente rotulada como de maior idade inclui participantes com exatamente 40 anos.

## Função retardada do enxerto

`dgf_main` deriva de `dial_1stweek`, indicador de diálise na primeira semana após o transplante. Codificações históricas alternativas não substituem essa fonte. Como a DGF é informação pós-transplante, sua interpretação é prognóstica e retrospectiva; a análise landmark no dia 7 avalia a sensibilidade ao desalinhamento temporal.

## Desfecho de sobrevivência

O evento é o óbito do receptor. O modelo de Cox principal utiliza o seguimento completo disponível; a sensibilidade administrativa encerra o acompanhamento em 30 de junho de 2015; o Cox landmark restringe a população elegível no dia 7 e reinicia a escala temporal depois desse marco.

## Desfechos binários

- óbito até o horizonte: evento;
- vida confirmada no horizonte: não evento;
- censura antes do horizonte sem óbito: outcome não observável.

Complete case é aplicado após a definição de observabilidade. O horizonte binário landmark permanece no dia 730 desde o transplante. Cinco anos é examinado apenas quanto à viabilidade; nenhum modelo binário de cinco anos integra o pipeline final.

## Calibração

O intercepto de calibração (`calibration-in-the-large`) avalia se, globalmente, as probabilidades previstas tendem a ficar sistematicamente acima ou abaixo da frequência observada dos eventos. Valores próximos de zero indicam ausência de deslocamento global importante. A estimativa utilizada corresponde a:

`logit(P(Y=1)) = alpha + offset(logit(p))`

Em termos gerais, intercepto positivo indica tendência de subestimação global do risco, enquanto intercepto negativo indica tendência de superestimação global do risco.

A inclinação de calibração avalia se a dispersão das probabilidades previstas é compatível com a observada. A estimativa corresponde a:

`logit(P(Y=1)) = alpha + beta * logit(p)`

Valores próximos de um indicam dispersão aproximadamente adequada. Em termos gerais, valores menores que um indicam previsões excessivamente extremas; valores maiores que um indicam previsões pouco extremas ou excessivamente concentradas.

As duas quantidades são estimadas separadamente e têm finalidade diagnóstica, especialmente diante do número reduzido de eventos. Elas não constituem critérios de validação externa ou de utilidade clínica.
