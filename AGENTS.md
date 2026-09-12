# AGENTS.md

## 1. Finalidade

Este arquivo contém regras de desenvolvimento e decisões metodológicas que devem ser respeitadas por qualquer agente de IA que trabalhe neste repositório.

Não alterar decisões metodológicas, equações, parâmetros experimentais ou critérios de avaliação sem autorização explícita.

Quando houver ambiguidade, preservar a implementação existente e indicar a dúvida antes de introduzir uma nova hipótese.

---

# 2. Contexto do projeto

O projeto investiga a otimização de rotas para coleta de leite utilizando dados de localização, temperatura e volume.

O foco desta etapa é exclusivamente computacional.

Não existe protótipo físico no escopo atual.

Os dados serão fornecidos ao backend por arquivos previamente gerados, prioritariamente em CSV ou JSON.

---

# 3. Formulação geral do problema

O problema utiliza:

- um único veículo;
- um único depósito/laticínio;
- rota em circuito fechado;
- início da rota no depósito;
- término da rota no mesmo depósito;
- todos os pontos de coleta devem ser atendidos;
- cada ponto deve ser visitado uma única vez;
- capacidade do veículo tratada como restrição rígida;
- as instâncias devem ser construídas de forma que todo o volume possa ser transportado;
- tempo de atendimento/coleta fixo por ponto.

A implementação deve rejeitar ou sinalizar rotas inválidas em vez de corrigi-las silenciosamente.

---

# 4. Cenários experimentais

As escalas principais são:

- 10 pontos;
- 25 pontos;
- 50 pontos.

Para cada escala devem ser consideradas:

- 10 instâncias independentes.

As meta-heurísticas devem ser executadas:

- 30 vezes por instância.

As execuções devem ser reproduzíveis.

SA e ACO utilizarão um conjunto previamente definido de 30 sementes aleatórias, uma para cada repetição experimental.

Caso existam reinicializações internas, as sementes derivadas devem partir da semente principal da repetição.

Não utilizar geração aleatória sem controle explícito de seed nos experimentos finais.

---

# 5. Algoritmos

Devem ser implementados separadamente:

1. Vizinho Mais Próximo clássico;
2. Vizinho Mais Próximo multicritério;
3. Simulated Annealing;
4. Ant Colony Optimization.

Os algoritmos devem compartilhar estruturas de dados e funções de avaliação sempre que possível.

Evitar duplicação de lógica entre algoritmos.

---

# 6. Vizinho Mais Próximo clássico

A versão clássica deve utilizar exclusivamente distância como critério para selecionar o próximo ponto.

Ela será utilizada como baseline.

A rota deve:

- começar no depósito;
- visitar todos os pontos exatamente uma vez;
- retornar ao depósito.

Não adicionar temperatura ou ocupação nesta versão.

---

# 7. Vizinho Mais Próximo multicritério

A versão multicritério utiliza a função de custo definida para o projeto.

Os critérios são:

- componente logístico;
- risco térmico;
- ocupação.

A implementação deve permanecer separada da versão clássica.

Não modificar silenciosamente a definição dos critérios ou suas normalizações.

---

# 8. Função objetivo multicritério

A estrutura geral utiliza três componentes:

```text
J = w_L * J_L + w_T * J_T + w_O * J_O
```

onde:

- `J_L` representa o componente logístico;
- `J_T` representa o componente térmico;
- `J_O` representa o componente de ocupação;
- `w_L`, `w_T` e `w_O` são os respectivos pesos.

Os componentes devem ser implementados separadamente antes de sua agregação.

Evitar valores hardcoded dentro dos algoritmos.

Os pesos devem vir de configuração.

---

# 9. Perfis de pesos

Perfis principais:

## Logístico

```text
logístico = 0.60
térmico   = 0.20
ocupação  = 0.20
```

## Equilibrado

```text
logístico = 1/3
térmico   = 1/3
ocupação  = 1/3
```

## Térmico

```text
logístico = 0.20
térmico   = 0.60
ocupação  = 0.20
```

Também será realizada análise de sensibilidade.

A análise deve considerar variações de:

```text
±0.10
±0.20
```

sobre o peso dominante, redistribuindo a diferença entre os outros critérios conforme a metodologia definida.

---

# 10. Risco térmico

O projeto utiliza medições de temperatura.

Não introduzir modelos microbiológicos ou químicos.

Não criar modelos preditivos de temperatura sem decisão metodológica explícita.

O sistema deve trabalhar com a medição disponível e com eventos relacionados ao comportamento térmico definido pelo estudo.

---

# 11. Ocupação

A ocupação representa o uso da capacidade do veículo.

Não assumir automaticamente que baixo volume significa situação ruim.

O termo de ocupação deve seguir a formulação metodológica estabelecida no projeto.

A capacidade do veículo é uma restrição rígida.

Uma solução que exceda a capacidade não deve ser considerada válida.

---

# 12. Recálculo de rota

A rota poderá ser recalculada quando ocorrer evento relevante.

Eventos previstos incluem:

- conclusão de uma coleta;
- evento térmico relevante;
- alteração relevante de ocupação.

Não implementar recálculo contínuo ou periódico sem evento.

Não criar novas regras de disparo sem decisão metodológica.

---

# 13. Simulated Annealing

O Simulated Annealing deve ser implementado como módulo independente.

Seus parâmetros serão definidos por testes-piloto para cada escala antes dos experimentos finais.

Após a calibração, os parâmetros selecionados devem permanecer congelados durante os experimentos finais.

A função objetivo deve ser compartilhada com os demais algoritmos multicritério.

A aleatoriedade deve utilizar a seed da repetição experimental.

---

# 14. Ant Colony Optimization

O ACO deve ser implementado como módulo independente.

Seus parâmetros também serão definidos por testes-piloto para cada escala e posteriormente congelados.

A atualização de feromônio deve utilizar a melhor formiga de cada iteração, conforme a formulação metodológica definida.

Não alterar a estratégia de atualização de feromônio sem autorização.

A aleatoriedade deve utilizar a seed da repetição experimental.

---

# 15. Estatística

Para comparações estatísticas foi definido:

```text
alpha = 0.05
```

Esse valor deve ser tratado como escolha metodológica convencional.

Não interpretar automaticamente `p > 0.05` como prova de igualdade entre algoritmos.

Não selecionar testes estatísticos automaticamente sem verificar previamente os pressupostos aplicáveis.

---

# 16. Organização do código

Responsabilidades devem ser separadas.

Estrutura esperada conceitualmente:

```text
src/
├── algorithms/
├── models/
├── objective/
├── simulation/
├── experiments/
└── utils/
```

A estrutura poderá evoluir conforme a implementação, mas evitar arquivos monolíticos.

Não colocar todos os algoritmos em um único arquivo.

---

# 17. Convenções Python

Utilizar:

- Python 3.11;
- `snake_case` para funções e variáveis;
- `PascalCase` para classes;
- type hints quando úteis;
- docstrings em funções públicas relevantes;
- funções pequenas e com responsabilidade clara;
- imports explícitos;
- caminhos tratados preferencialmente com `pathlib`.

Evitar:

- variáveis globais mutáveis;
- números mágicos;
- duplicação de código;
- funções excessivamente grandes;
- lógica experimental misturada com visualização;
- efeitos colaterais inesperados em imports.

---

# 18. Configuração

Parâmetros experimentais devem ficar fora da lógica principal sempre que possível.

Utilizar a pasta:

```text
config/
```

para parâmetros como:

- perfis de pesos;
- escalas;
- seeds;
- quantidade de execuções;
- parâmetros calibrados de SA;
- parâmetros calibrados de ACO.

Não espalhar esses valores diretamente pelos algoritmos.

---

# 19. Testes

Toda funcionalidade crítica deve ser acompanhada de testes.

Exemplos obrigatórios ao longo do desenvolvimento:

- rota começa no depósito;
- rota termina no depósito;
- todos os pontos são visitados;
- nenhum ponto é visitado duas vezes;
- capacidade é respeitada;
- custo de rota é calculado corretamente;
- mesma seed produz mesmo resultado;
- instâncias inválidas são detectadas.

Os testes devem ficar em:

```text
tests/
```

Executar:

```text
pytest
```

antes de considerar uma funcionalidade concluída.

---

# 20. Qualidade do código

Antes de concluir uma tarefa relevante:

```text
ruff check .
black --check .
pytest
```

Quando necessário, formatar com:

```text
black .
```

Não alterar configuração de lint ou formatação apenas para esconder erros.

---

# 21. Resultados experimentais

Dados de entrada, código e resultados devem permanecer separados.

Utilizar:

```text
data/
results/
```

Não sobrescrever resultados anteriores silenciosamente.

Resultados finais devem incluir metadados suficientes para identificar:

- algoritmo;
- instância;
- escala;
- seed;
- perfil de pesos;
- parâmetros utilizados;
- métricas obtidas.

---

# 22. Reprodutibilidade

A reprodutibilidade é requisito central.

Toda execução experimental deve permitir identificar:

- versão do código;
- configuração;
- seed;
- instância;
- algoritmo;
- parâmetros.

Não utilizar estado aleatório global sem controle.

---

# 23. Uso do Git

Antes de mudanças grandes:

1. verificar `git status`;
2. trabalhar em alterações pequenas e coerentes;
3. executar testes;
4. revisar o diff;
5. realizar commit descritivo.

Não executar operações destrutivas de Git como:

```text
reset --hard
push --force
rebase
```

sem solicitação explícita.

---

# 24. Comportamento esperado do agente

Antes de implementar uma funcionalidade importante:

1. ler este arquivo;
2. analisar os módulos relacionados;
3. identificar arquivos que precisarão ser alterados;
4. preservar interfaces existentes sempre que possível;
5. implementar a menor mudança necessária;
6. adicionar ou atualizar testes;
7. executar os testes relevantes;
8. informar qualquer decisão que não esteja definida neste documento.

Não modificar a metodologia para facilitar a implementação.

Não inventar parâmetros.

Não alterar equações sem autorização.

Não adicionar bibliotecas quando a biblioteca padrão ou dependências existentes resolverem adequadamente o problema.

---

# 25. Prioridade das instruções

Em caso de conflito, utilizar a seguinte prioridade:

1. decisão explícita mais recente fornecida pelo autor do projeto;
2. documentação metodológica do TCC;
3. este `AGENTS.md`;
4. implementação existente;
5. inferência do agente.

Quando houver conflito entre documentos, sinalizar o conflito antes de modificar o comportamento do sistema.
