# TCC — Roteirização Inteligente para Coleta de Leite

Projeto computacional desenvolvido como parte do Trabalho de Conclusão de Curso, com foco na otimização de rotas de coleta de leite considerando critérios logísticos, térmicos e de ocupação do veículo.

## Objetivo

Desenvolver e comparar estratégias de roteirização capazes de determinar rotas de coleta considerando:

- distância e tempo de deslocamento;
- risco térmico associado às medições de temperatura;
- ocupação/capacidade do veículo;
- eventos que possam justificar o recálculo da rota durante a operação.

## Algoritmos

O estudo considera:

- Vizinho Mais Próximo clássico;
- Vizinho Mais Próximo multicritério;
- Simulated Annealing;
- Ant Colony Optimization.

## Estrutura do projeto

```text
tcc-roteirizacao-leite/
│
├── config/        # Configurações dos experimentos
├── data/          # Instâncias e dados de entrada
├── docs/          # Documentação técnica
├── notebooks/     # Análises exploratórias e resultados
├── results/       # Saídas dos experimentos
├── src/           # Código-fonte
├── tests/         # Testes automatizados
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

Versão de Python utilizada:

```text
Python 3.11
```

### Criar ambiente virtual

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Instalar ferramentas de desenvolvimento

```powershell
python -m pip install -r requirements-dev.txt
```

As dependências computacionais específicas do projeto serão registradas em:

```text
requirements.txt
```

## Ferramentas de desenvolvimento

O projeto utiliza:

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

## Controle de versão

O projeto utiliza Git e um repositório privado no GitHub.

Fluxo básico:

```powershell
git status
git add .
git commit -m "Descrição da alteração"
git push
```

## Status

O projeto encontra-se na etapa de desenvolvimento computacional, após a definição do escopo, revisão teórica, formulação matemática e parametrização computacional.
