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

## 3.1 Rede viária

A rede viária é uma estrutura independente dos pontos de coleta. A relação deve ser feita por `node_id`:

```text
CollectionPoint.node_id ──> RoadNetwork.nodes
Depot.node_id            ──> RoadNetwork.nodes
```

As regras da rede são:

- `node_id`, `point_id` e `depot_id` são inteiros e pertencem a espaços de identificação independentes;
- cada ponto de coleta ocupa exatamente um nó da rede;
- dois pontos de coleta não podem compartilhar `node_id`;
- o depósito ocupa um nó exclusivo e não pode compartilhar nó com ponto de coleta;
- a rede sintética é dirigida e deve ser fortemente conexa;
- os arcos `i -> j` e `j -> i` podem existir independentemente;
- não são permitidos múltiplos arcos com o mesmo par ordenado `(i, j)`;
- `RoadArc` armazena distância em metros;
- o tempo de viagem não deve ser armazenado no arco;
- o tempo deve ser calculado por `travel_time = distance / v_ref`;
- `v_ref` pertence à `ProblemInstance`, é expresso em m/s e é constante para a instância;
- coordenadas cartesianas dos nós da malha sintética são expressas em metros.

Um caminho inexistente em uma instância aceita representa erro de consistência e deve gerar `ValueError`.

As coordenadas dos nós podem ser usadas para geração da malha e visualização, mas os deslocamentos devem respeitar exclusivamente os arcos dirigidos da rede. Não calcular distância operacional diretamente entre coordenadas dos pontos de coleta.

## 3.2 Unidades internas

As unidades internas são:

- distância: metros;
- velocidade: m/s;
- tempo: segundos;
- volume e capacidade: litros;
- temperatura: °C.

Conversões de unidade devem ocorrer somente para apresentação, quando necessário.

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

Na Formulação V1, para cada ponto `i`, `T_i(t)` representa a temperatura do leite disponível no ponto no instante `t`.

- enquanto o ponto não for coletado, o valor evolui conforme uma trajetória discreta pré-gerada;
- no instante da coleta, a temperatura vigente deve ser registrada;
- após a coleta, o ponto deixa de ser atualizado;
- não modelar a evolução térmica do leite depois que ele entra no veículo;
- as trajetórias pertencem à `ProblemInstance`, e não permanentemente a `CollectionPoint`.

---

# 11. Ocupação

A ocupação representa o uso da capacidade do veículo.

Não assumir automaticamente que baixo volume significa situação ruim.

O termo de ocupação deve seguir a formulação metodológica estabelecida no projeto.

A capacidade do veículo é uma restrição rígida.

Uma solução que exceda a capacidade não deve ser considerada válida.

Na geração da instância deve valer:

```text
sum(max_t(q_i(t))) <= vehicle_capacity
```

Assim, qualquer ordenação válida dos pontos permanece viável quanto à capacidade.

Para cada ponto `i`, `q_i(t)` representa o volume de leite disponível para coleta no instante `t`.

- enquanto o ponto não for atendido, `q_i(t)` evolui conforme uma trajetória discreta pré-gerada;
- no instante da coleta, o volume vigente é adicionado à carga do veículo;
- após a coleta, o ponto é considerado atendido;
- as trajetórias pertencem à `ProblemInstance`, e não permanentemente a `CollectionPoint`.

---

# 12. Recálculo de rota

A rota poderá ser recalculada quando ocorrer evento relevante.

Eventos previstos incluem:

- conclusão de uma coleta;
- evento térmico relevante;
- alteração relevante de ocupação.

Não implementar recálculo contínuo ou periódico sem evento.

Não criar novas regras de disparo sem decisão metodológica.

## 12.1 Amostragem, horizonte e eventos

Temperatura e volume devem ser atualizados a cada 30 segundos, utilizando sample-and-hold:

- entre duas amostras vale a última amostra disponível;
- não realizar interpolação na V1;
- eventos relacionados a temperatura ou volume só podem surgir quando chegar uma nova amostra;
- as trajetórias devem ter horizonte suficiente para cobrir toda a execução;
- ultrapassar o horizonte disponível é erro de configuração;
- não extrapolar nem manter silenciosamente a última amostra indefinidamente.

Eventos térmicos devem ser gerados somente quando houver mudança de faixa térmica. Permanecer na mesma faixa em amostras sucessivas não gera novo evento. Os valores numéricos dos limites térmicos serão parametrizados posteriormente.

Eventos de ocupação devem ser gerados somente quando houver mudança entre faixas discretas de ocupação. Pequenas variações dentro da mesma faixa não geram evento. Os limites numéricos dessas faixas serão definidos posteriormente.

## 12.2 Ordem de processamento de eventos

A ordem de processamento deve ser:

1. avançar o tempo;
2. atualizar temperatura e volume para a última amostra disponível;
3. aplicar o evento ocorrido;
4. atualizar o estado da simulação;
5. avaliar os gatilhos de recálculo;
6. recalcular a rota, se necessário;
7. continuar a execução.

Se um gatilho surgir enquanto o veículo atravessa um arco:

- não interromper o arco;
- a nova rota passa a valer quando o veículo alcançar o próximo nó;
- não modelar posição contínua dentro do arco na V1.

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

Os modelos de domínio mínimos previstos em `src/models/` são:

- `RoadNode`;
- `RoadArc`;
- `RoadNetwork`;
- `CollectionPoint`;
- `Depot`;
- `ProblemInstance`;
- `Route`;
- uma representação simples de parada equivalente a `RouteStop(kind, entity_id)`.

`RoadNetwork` deve modelar nós e arcos dirigidos sem armazenar pontos de coleta. `CollectionPoint` e `Depot` devem referenciar a rede por `node_id`.

A `Route` armazena somente a sequência lógica de paradas. As paradas devem distinguir explicitamente depósito de ponto de coleta. Caminhos expandidos sobre a rede são derivados quando necessário. Evitar hierarquias ou abstrações desnecessárias.

A separação do estado mutável da execução está definida na seção "Estado dinâmico da simulação".

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
- `v_ref` de cada instância;
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

Devem ser separados:

- `instance_seed`: seed utilizada na geração da instância;
- `algorithm_seed`: seed utilizada na aleatoriedade das execuções de SA e ACO.

As 30 repetições das meta-heurísticas devem utilizar um conjunto fixo e previamente definido de `algorithm_seed`. Sementes derivadas de reinicializações internas devem partir da `algorithm_seed` da repetição.

Os metadados essenciais da instância e da execução devem ser persistidos em JSON.

## Estado dinâmico da simulação

A `ProblemInstance` representa o cenário e permanece imutável durante a execução. O estado mutável deve futuramente ficar em `SimulationState`, dentro de `src/simulation/`.

Esse estado poderá conter, entre outros:

- tempo atual;
- nó atual;
- carga atual;
- pontos atendidos;
- pontos restantes;
- rota ativa;
- valores correntes de temperatura e volume.

Não implementar `SimulationState` ainda.

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

Utilizar `ValueError` com mensagens claras para violações de domínio nesta etapa. Não criar uma hierarquia própria de exceções enquanto não houver necessidade concreta.

As estruturas devem permanecer simples, explícitas, tipadas e fáceis de explicar academicamente. Não introduzir complexidade arquitetural sem necessidade concreta.

---

# 25. Prioridade das instruções

Em caso de conflito, utilizar a seguinte prioridade:

1. decisão explícita mais recente fornecida pelo autor do projeto;
2. documentação metodológica do TCC;
3. este `AGENTS.md`;
4. implementação existente;
5. inferência do agente.

Quando houver conflito entre documentos, sinalizar o conflito antes de modificar o comportamento do sistema.
