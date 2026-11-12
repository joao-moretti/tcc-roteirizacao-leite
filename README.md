# TCC — Roteirização Inteligente para Coleta de Leite

Projeto computacional desenvolvido como parte do Trabalho de Conclusão de Curso em Engenharia de Controle e Automação, com foco na comparação de estratégias de roteirização para coleta de leite em um problema dinâmico e multicritério.

O sistema considera simultaneamente aspectos logísticos, térmicos e volumétricos, permitindo o recálculo da rota a partir de mudanças observadas durante a operação.

## Objetivo

Desenvolver, implementar e comparar estratégias de roteirização capazes de determinar a sequência de atendimento dos produtores considerando:

- distância e tempo de deslocamento;
- risco térmico associado à temperatura do leite;
- permanência do volume de leite nos pontos de coleta;
- atualização dinâmica das condições dos produtores;
- eventos que possam justificar a reotimização da rota.

A avaliação das rotas é realizada por uma função objetivo multicritério baseada em três componentes:

\[
J = w_D\bar{J}_D + w_T\bar{J}_T + w_O\bar{J}_O
\]

em que:

- \(J_D\) representa o desempenho logístico;
- \(J_T\) representa o custo associado ao risco térmico;
- \(J_O\) representa o custo associado à permanência volumétrica;
- \(w_D\), \(w_T\) e \(w_O\) representam os pesos atribuídos a cada critério.

## Algoritmos

O estudo compara quatro estratégias de roteirização:

- Vizinho Mais Próximo clássico (`NN`);
- Vizinho Mais Próximo multicritério (`MCNN`);
- Simulated Annealing (`SA`);
- Ant Colony Optimization (`ACO`).

O Vizinho Mais Próximo clássico é utilizado como baseline logístico, enquanto MCNN, SA e ACO incorporam os critérios multicritério definidos no modelo.

## Modelo experimental

Os experimentos utilizam instâncias sintéticas com três escalas:

- 5 pontos de coleta;
- 15 pontos de coleta;
- 20 pontos de coleta.

A rede é representada por um grafo dirigido, contendo uma combinação parametrizada de arcos bidirecionais e unidirecionais.

Para cada instância, as distâncias \(d_{ij}\) são obtidas pelo menor caminho disponível na rede. O tempo de deslocamento é calculado por:

\[
t_{ij}=\frac{d_{ij}}{v_{\mathrm{ref}}}
\]

em que \(v_{\mathrm{ref}}\) é mantida constante dentro da instância.

A operação considera:

- um único veículo;
- um único depósito;
- rota iniciando e terminando no depósito;
- capacidade suficiente para o volume total;
- tempo de serviço fixo;
- trajetórias térmicas e volumétricas pré-geradas e congeladas;
- reotimização orientada a eventos.

## Perfis multicritério

São utilizados três perfis principais de pesos:

| Perfil | Logístico | Térmico | Volumétrico |
|---|---:|---:|---:|
| Logístico | 0,60 | 0,20 | 0,20 |
| Equilibrado | 1/3 | 1/3 | 1/3 |
| Térmico | 0,20 | 0,60 | 0,20 |

Os mesmos cenários e condições experimentais são utilizados entre algoritmos comparáveis para preservar a reprodutibilidade das análises.

## Estrutura do projeto

```text
tcc-roteirizacao-leite/
│
├── docs/                    # Documentação e registros metodológicos
├── results/                 # Resultados e artefatos experimentais
├── src/                     # Código-fonte do projeto
│   ├── algorithms/          # NN, MCNN, SA e ACO
│   ├── experiments/         # Execução e configuração dos experimentos
│   ├── models/              # Estruturas e modelos do problema
│   ├── objective/           # Função objetivo e critérios de avaliação
│   ├── simulation/          # Dinâmica e reotimização
│   └── utils/               # Funções auxiliares
│
├── tests/                   # Testes automatizados
│
├── .vscode/
├── .gitignore
├── AGENTS.md
├── pyproject.toml
├── README.md
├── requirements.txt
└── requirements-dev.txt
```

## Ambiente

Versão utilizada no desenvolvimento:

```text
Python 3.11.9
```

### Criar ambiente virtual

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Instalar dependências

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

## Ferramentas de desenvolvimento

O projeto utiliza:

- `NumPy` para operações numéricas;
- `Pandas` para organização e análise de dados;
- `pytest` para testes automatizados;
- `ruff` para análise estática e lint;
- `black` para formatação do código.

### Executar testes

```powershell
pytest
```

### Verificar código

```powershell
ruff check .
```

### Formatar código

```powershell
black .
```

## Reprodutibilidade

A implementação foi estruturada para permitir repetição controlada dos experimentos.

Entre os mecanismos adotados estão:

- sementes pseudoaleatórias fixadas;
- mesmas instâncias para algoritmos comparáveis;
- trajetórias de temperatura e volume congeladas;
- separação entre geração dos cenários e execução dos algoritmos;
- registro dos parâmetros experimentais;
- testes automatizados para componentes críticos do modelo.

SA e ACO utilizam sementes controladas para permitir análise da variabilidade entre execuções.

## Campanha experimental

A campanha experimental oficial foi concluída para as três escalas.

O arquivo:

```text
results/official_runs.jsonl
```

constitui a base experimental oficial utilizada nas análises.

A campanha contém:

```text
11.040 execuções
```

distribuídas igualmente entre as escalas de 5, 15 e 20 pontos:

```text
3.680 execuções por escala
```

Os resultados incluem execuções em condições nominais e com perturbações, permitindo avaliar desempenho, robustez e comportamento dinâmico dos algoritmos.

## Métricas analisadas

Entre as principais métricas consideradas estão:

- distância total percorrida;
- tempo total da operação;
- temperatura média observada;
- temperatura máxima observada;
- comportamento sob perturbações;
- número de chamadas adicionais de otimização;
- interrupções e eventos de reotimização;
- variabilidade entre sementes;
- custo computacional.

A comparação dos algoritmos é realizada de forma multicritério, sem assumir a existência de um único algoritmo superior em todas as métricas.

## Controle de versão

O projeto utiliza Git e possui repositório no GitHub.

Fluxo básico:

```powershell
git status
git add .
git commit -m "Descrição da alteração"
git push
```

## Status atual

As etapas de definição do problema, revisão teórica, formulação matemática, especificação computacional, desenvolvimento e execução experimental foram concluídas.

A campanha oficial de **11.040 execuções** também foi concluída e passou pelas verificações de integridade e sanidade.

O projeto encontra-se atualmente na etapa de **análise e discussão dos resultados experimentais**, envolvendo:

- comparação entre NN, MCNN, SA e ACO;
- avaliação das três escalas experimentais;
- comparação entre cenários nominal e `disturbance`;
- influência dos diferentes perfis de pesos;
- análise da variabilidade entre execuções;
- avaliação do comportamento dinâmico;
- análise do custo computacional;
- consolidação dos resultados para o texto final do TCC.

O desenvolvimento dos algoritmos e os dados da campanha oficial são considerados congelados, salvo identificação de erro metodológico que comprometa a validade dos experimentos.
