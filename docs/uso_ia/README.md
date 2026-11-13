# Registro do uso de inteligência artificial no TCC

> **Status: documentação em elaboração.** Estes registros não constituem, por si só, a declaração final de uso de IA nem substituem as exigências da PUC Campinas e a orientação do professor.

## Finalidade

Documentar de forma rastreável a utilização do ChatGPT (OpenAI) como ferramenta auxiliar durante o TCC de roteirização dinâmica da coleta de leite, distinguindo apoio na redação acadêmica, pesquisa e compreensão de conceitos, formulação matemática, planejamento, programação, testes e análise de resultados.

O autor permanece responsável pelas decisões metodológicas, pela implementação, pela verificação dos resultados e pelo conteúdo final apresentado.

## Organização

- [Inventário das conversas](inventario_conversas.md): títulos e classificação preliminar das conversas usadas no projeto.
- [Modelo de registro](modelo_registro.md): campos a preencher para cada conversa; somente comandos efetivamente recuperados devem ser transcritos.
- [Manifesto de scripts](manifesto_scripts.md): índice dos arquivos de código cuja elaboração ou revisão contou com auxílio de IA, acompanhado de indicação dos testes e da versão do repositório.

Os registros específicos das conversas deverão ser adicionados posteriormente nesta pasta, em arquivos Markdown distintos ou por agrupamento temático, mantendo vínculos com o inventário.

## Procedimento de preenchimento

1. Identificar a conversa, seu período e o(s) modelo(s) utilizado(s), **somente quando for possível confirmar a informação no histórico**. Caso contrário, registrar “versão não identificada”.
2. Transcrever os comandos (*prompts*) relevantes e efetivamente enviados, sem criar reconstruções apresentadas como transcrições literais.
3. Relacionar as respostas ou materiais aproveitados e identificar a seção do TCC à qual contribuíram.
4. Para códigos, apontar os arquivos existentes no repositório, a natureza do auxílio e as verificações realizadas; preservar uma versão identificável do projeto para a entrega.
5. Revisar as informações antes de publicar. **Este repositório é público**: não incluir dados pessoais desnecessários, credenciais, mensagens privadas nem exportações integrais não revisadas.
6. Preparar o apêndice do TCC com resumo por etapa, identificação da ferramenta e referência aos registros complementares. Confirmar com o orientador se o acesso ao GitHub atende às exigências de apresentar todos os *scripts* e/ou comandos.

## Distinção importante

- **Prompt/comando**: texto enviado à ferramenta de IA, relacionado à tarefa acadêmica.
- **Script**: código-fonte, por exemplo, arquivos Python em `src/`, `tests/` e `notebooks/`. Um arquivo estar no repositório **não prova**, por si só, que foi produzido com auxílio da IA.
- **Artefato final**: texto, tabela, gráfico, teste ou código efetivamente incorporado ao TCC.

A atribuição de contribuições à IA deve seguir os registros reais; não marcar indiscriminadamente todo o código como “gerado por IA”.
