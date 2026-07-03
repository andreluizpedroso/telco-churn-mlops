# Plano de Sprints

Este projeto sera feito em etapas pequenas. A ideia e aprender o fluxo completo de um projeto de ML sem tentar resolver tudo de uma vez.

## Sprint 1: Fundacao do projeto

### Objetivo

Criar a base do repositorio e deixar claro qual problema sera resolvido.

### Explicacao para junior

Antes de treinar qualquer modelo, precisamos organizar o projeto. Em empresas, outras pessoas precisam conseguir abrir o repositorio, entender o objetivo, instalar dependencias e encontrar cada parte do codigo.

Um repositorio bem organizado evita que notebooks, dados, modelos e scripts fiquem misturados.

### Entregas

- Estrutura de pastas do projeto.
- README inicial.
- `.gitignore` para Python e artefatos de ML.
- `pyproject.toml` com dependencias planejadas.
- Documentacao das sprints.
- Perguntas e respostas de estudo.

### Criterio de pronto

A Sprint 1 esta pronta quando o repositorio possui estrutura clara e documentacao suficiente para uma pessoa iniciante entender o que sera construido.

## Sprint 2: EDA e baselines

### Objetivo

Entender os dados e criar modelos simples de referencia.

### Explicacao para junior

EDA significa Exploratory Data Analysis. Nessa etapa, olhamos os dados para responder perguntas como: existem valores ausentes? A classe churn esta balanceada? Quais variaveis parecem importantes?

Depois criamos baselines. Um baseline e um modelo simples que serve como comparacao. Se uma rede neural nao superar um baseline, talvez ela nao seja a melhor escolha.

### Entregas

- Notebook de EDA.
- Tratamento inicial dos dados.
- Separacao treino, validacao e teste.
- Baseline com `DummyClassifier`.
- Baseline com Regressao Logistica.
- Registro dos experimentos com MLflow.

## Sprint 3: Rede neural MLP com PyTorch

### Objetivo

Construir, treinar e avaliar uma rede neural MLP.

### Explicacao para junior

MLP significa Multilayer Perceptron. Ela e uma rede neural formada por camadas densas. Para dados tabulares, como dados de clientes, ela recebe as features de entrada e devolve uma probabilidade de churn.

A rede precisa ser comparada com os baselines. O objetivo nao e apenas criar uma rede neural, mas provar se ela ajuda ou nao.

### Entregas

- Modelo MLP em PyTorch.
- Treinamento com seed fixa.
- Validacao estratificada.
- Early stopping.
- Comparacao com baselines.
- Registro no MLflow.

## Sprint 4: Pipeline, testes e API

### Objetivo

Transformar os experimentos em codigo reutilizavel e disponibilizar inferencia por API.

### Explicacao para junior

Notebook e bom para descobrir coisas. Codigo em `src/` e melhor para producao. Nesta sprint, a gente separa preprocessing, treino, avaliacao e predicao em modulos testaveis.

Depois criamos uma API com FastAPI. A API recebe dados de um cliente e responde a probabilidade de churn.

### Entregas

- Codigo modular em `src/`.
- Pipeline de preprocessing.
- Testes com `pytest`.
- Endpoint `/health`.
- Endpoint `/predict`.
- Validacao com Pydantic.
- Logging estruturado.

## Sprint 5: Documentacao final e entrega

### Objetivo

Preparar o projeto para avaliacao e explicar as decisoes tecnicas.

### Explicacao para junior

Um projeto de ML nao termina quando o modelo treina. Precisamos documentar resultados, limitacoes, riscos, vieses e como monitorar o modelo depois que ele estiver em uso.

Tambem precisamos preparar o video final usando STAR: Situation, Task, Action e Result.

### Entregas

- README final.
- Model Card.
- Plano de monitoramento.
- Documentacao da arquitetura.
- Roteiro STAR.
- Opcional: deploy em nuvem.
