# Disponibilidade dos dados

Os dados individuais utilizados no trabalho não são redistribuídos neste repositório. Também não se incluem bases derivadas linha a linha, identificadores, datas clínicas individuais, previsões OOF individuais ou registros de bootstrap por participante ou tentativa.

Os scripts correspondem às análises descritas no trabalho, mas a reprodução integral depende das entradas analíticas correspondentes. O manuscrito informa que os dados secundários anonimizados utilizados no trabalho foram disponibilizados publicamente em associação ao artigo-base. Este repositório não estabelece nem garante uma URL ou DOI específico para acesso aos dados; recomenda-se consultar o artigo-base e seus materiais associados.

Em nível estrutural, as rotinas esperam:

- uma coorte bruta em CSV com as variáveis de origem utilizadas pela preparação;
- bases processadas de sobrevivência com tempo de acompanhamento, indicador de óbito e preditores definidos no estudo;
- bases binárias observáveis para um ano, dois anos e a sensibilidade landmark de dois anos;
- um identificador técnico por participante para preservar a unidade de reamostragem e OOF.

Esse identificador é necessário à execução local autorizada, mas nenhum valor individual é distribuído. Diretórios de dados brutos e processados são bloqueados pelo `.gitignore`.
