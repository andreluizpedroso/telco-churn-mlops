# Telco Churn MLOps

Projeto do Tech Challenge Fase 01 para prever churn de clientes de telecomunicacoes com uma rede neural MLP em PyTorch, comparacao com baselines em Scikit-Learn, rastreamento de experimentos com MLflow e API de inferencia com FastAPI.

## Contexto

Uma operadora de telecomunicacoes esta perdendo clientes em ritmo acelerado. A diretoria solicitou um modelo preditivo que classifique clientes com risco de cancelamento para apoiar acoes de retencao.

O projeto cobre o fluxo completo: entendimento do problema, analise exploratoria, treinamento de modelos, comparacao de resultados, empacotamento do pipeline, API de inferencia, testes e documentacao final.

## Objetivos

- Analisar um dataset publico de churn em telecomunicacoes.
- Criar baselines com Scikit-Learn.
- Treinar uma rede neural MLP com PyTorch.
- Comparar modelos usando metricas tecnicas e de negocio.
- Registrar experimentos, parametros, metricas e artefatos com MLflow.
- Disponibilizar inferencia por API com FastAPI.
- Documentar limitacoes, vieses, arquitetura e plano de monitoramento.

## Estrutura do Repositorio

```text
.
|-- data/
|   |-- raw/          # Dados originais
|   `-- processed/    # Dados tratados
|-- docs/             # Documentacao, sprints, Model Card e roteiro STAR
|-- models/           # Modelos e artefatos treinados
|-- notebooks/        # EDA e experimentos exploratorios
|-- src/              # Codigo reutilizavel do projeto
`-- tests/            # Testes automatizados
```

## Requisitos Do Tech Challenge

- Repositorio GitHub organizado.
- README com instrucoes de setup, execucao e descricao do projeto.
- `pyproject.toml` com configuracao do projeto.
- Historico de commits claro e progressivo.
- `.gitignore` adequado para artefatos de ML.
- Seeds fixas para reproducibilidade.
- Validacao estratificada.
- Baselines com Scikit-Learn.
- Rede neural MLP treinada com PyTorch.
- Experimentos rastreados com MLflow.
- API FastAPI com endpoints `/health` e `/predict`.
- Validacao de entrada com Pydantic.
- Logging estruturado.
- Testes automatizados.
- Linting com `ruff`.
- Model Card com performance, limitacoes, vieses e cenarios de falha.
- Plano de monitoramento.
- Video final de ate 5 minutos no metodo STAR.

## Dataset

A base sugerida e um dataset publico de churn em telecomunicacoes, como o Telco Customer Churn. A fonte final, o dicionario de dados e as decisoes de tratamento serao documentados durante a etapa de EDA.

## Metricas

Metricas tecnicas planejadas:

- ROC-AUC
- F1-score
- Precision
- Recall
- Matriz de confusao

Metricas de negocio planejadas:

- Custo estimado de falso positivo.
- Custo estimado de falso negativo.
- Estimativa de churn evitado.
- Comparacao de trade-off entre reter clientes e evitar campanhas desnecessarias.

## Setup

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Instale o projeto com dependencias de desenvolvimento:

```bash
pip install -e ".[dev]"
```

## Comandos

Executar testes:

```bash
pytest
```

Executar lint:

```bash
ruff check .
```

Subir a API localmente:

```bash
uvicorn src.telco_churn_mlops.api:app --reload
```

## Plano De Entrega

1. Entendimento do problema, EDA e baselines.
2. Treinamento da MLP com PyTorch e comparacao de modelos.
3. Refatoracao em pipeline reproduzivel, testes e API.
4. Documentacao final, Model Card, plano de monitoramento e video STAR.

## Entregaveis

- Notebook de EDA.
- Modelos baseline.
- MLP treinada.
- Registros de experimentos no MLflow.
- API de inferencia.
- Testes passando.
- Documentacao tecnica.
- Model Card.
- Roteiro STAR para apresentacao final.
- Deploy em nuvem, caso seja realizado como bonus.
