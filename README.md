# Telco Churn MLOps

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-MLP-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Baselines-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Inference-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-EDA-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-Tests-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)
![Ruff](https://img.shields.io/badge/Ruff-Linting-D7FF64?style=for-the-badge&logo=ruff&logoColor=black)

Projeto do Tech Challenge Fase 01 para prever churn de clientes de telecomunicacoes com uma rede neural MLP em PyTorch, comparacao com baselines em Scikit-Learn, rastreamento de experimentos com MLflow e API de inferencia com FastAPI.

## Contexto

Uma operadora de telecomunicacoes esta perdendo clientes em ritmo acelerado. A diretoria solicitou um modelo preditivo que classifique clientes com risco de cancelamento para apoiar acoes de retencao.

O projeto cobre o fluxo completo: entendimento do problema, analise exploratoria, treinamento de modelos, comparacao de resultados, empacotamento do pipeline, API de inferencia, testes e documentacao final.

## Status Atual

| Item | Status |
| --- | --- |
| Estrutura do repositorio | Concluida |
| EDA e baselines | Concluidos |
| API FastAPI | Concluida com baseline de Regressao Logistica |
| MLP PyTorch | Implementada; treino real depende de PyTorch no ambiente ativo |
| MLflow | Pendente de execucao no ambiente ativo |
| Documentacao final | Publicada em `docs/` com os documentos exigidos pela entrega |

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
|-- docs/             # Documentacao final publicada
|-- models/           # Modelos e artefatos treinados
|-- notebooks/        # EDA e experimentos exploratorios
|-- src/              # Codigo reutilizavel do projeto
`-- tests/            # Testes automatizados
```

Observacao: somente a documentacao final da entrega e versionada em `docs/`. Anotacoes internas de sprint e estudo continuam locais e ignoradas pelo Git.

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

Metricas tecnicas usadas:

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

## Resultados Do Baseline

Modelo: Regressao Logistica com `class_weight="balanced"`.

| Metrica | Valor |
| --- | ---: |
| Accuracy | 0,726 |
| ROC-AUC | 0,835 |
| F1 | 0,607 |
| Precision | 0,490 |
| Recall | 0,797 |

O baseline identifica cerca de 79,7% dos clientes que realmente deram churn no conjunto de teste. Esse resultado serve como referencia para comparacao com a MLP.

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
$env:PYTHONPATH="src"
python -m telco_churn_mlops.train_baseline_model
uvicorn telco_churn_mlops.api:app --reload
```

Treinar a MLP:

```bash
$env:PYTHONPATH="src"
python -m telco_churn_mlops.train_mlp
```

Exemplo de payload para `/predict`:

```json
{
  "gender": "Female",
  "senior_citizen": 0,
  "partner": "Yes",
  "dependents": "No",
  "tenure": 12,
  "phone_service": "Yes",
  "multiple_lines": "No",
  "internet_service": "Fiber optic",
  "online_security": "No",
  "online_backup": "Yes",
  "device_protection": "No",
  "tech_support": "No",
  "streaming_tv": "Yes",
  "streaming_movies": "Yes",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check",
  "monthly_charges": 89.1,
  "total_charges": 950.5
}
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

## Documentacao Final

- [Model Card](docs/model-card.md)
- [Plano de monitoramento](docs/monitoramento.md)
- [Arquitetura da solucao](docs/arquitetura.md)
- [Roteiro STAR](docs/roteiro-star.md)
- [Checklist final](docs/checklist-final.md)
