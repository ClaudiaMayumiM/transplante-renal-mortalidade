# Ambiente computacional congelado

O manuscrito registra:

| Componente | Versão |
|---|---|
| Python | 3.12.7 |
| NumPy | 1.26.4 |
| pandas | 2.2.2 |
| scikit-learn | 1.8.0 |
| statsmodels | 0.14.2 |
| R | 4.3.1 |
| survival | 3.5.5 |

Os arquivos `python_requirements.txt` e `r_requirements.md` registram as dependências disponíveis. Também foi identificada a versão 2.2.1 de `svglite`. Os scripts de reporting usam `ggplot2`, `gridExtra`, `ragg`, `svglite` e `digest`, mas nem todas as versões históricas desses pacotes estavam fixadas nas evidências disponíveis; versões ausentes não foram inferidas.

Este repositório não atualiza dependências nem cria automaticamente um ambiente diferente do descrito. A instalação e a execução ficam fora do escopo desta documentação.
