# Telco Churn MLOps

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-MLP-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Baselines-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Inference-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-EDA-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-Tests-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)
![Ruff](https://img.shields.io/badge/Ruff-Linting-D7FF64?style=for-the-badge&logo=ruff&logoColor=black)

Projeto para prever churn de clientes de telecomunicacoes com uma rede neural MLP em PyTorch, comparacao com baselines em Scikit-Learn, rastreamento de experimentos com MLflow e API de inferencia com FastAPI.

## Contexto

Uma operadora de telecomunicacoes esta perdendo clientes em ritmo acelerado. A diretoria solicitou um modelo preditivo que classifique clientes com risco de cancelamento para apoiar acoes de retencao.

O projeto cobre o fluxo completo: entendimento do problema, analise exploratoria, treinamento de modelos, comparacao de resultados, empacotamento do pipeline, API de inferencia, testes e documentacao final.

## Status Atual

| Item | Status |
| --- | --- |
| Estrutura do repositorio | Concluida |
| EDA e baselines | Concluidos |
| Visualizacoes de EDA | Publicadas em `reports/figures/` |
| API FastAPI | Concluida com baseline de Regressao Logistica |
| MLP PyTorch | Treinada localmente com PyTorch CPU |
| MLflow | Tracking executado localmente com parametros, metricas e artefatos |
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
|   |-- raw/                         # Dados originais locais
|   `-- processed/                   # Dados tratados gerados pelo pipeline
|-- docs/                            # Documentacao final publicada
|   |-- arquitetura.md
|   |-- checklist-final.md
|   |-- model-card.md
|   |-- monitoramento.md
|   `-- roteiro-star.md
|-- models/                          # Artefatos locais gerados pelo treino
|-- notebooks/
|   `-- 01_eda_baselines.ipynb       # EDA e baselines
|-- reports/
|   `-- figures/                     # Graficos finais gerados pela EDA
|       |-- 01_churn_distribution.png
|       |-- 02_churn_by_contract.png
|       |-- 03_churn_by_payment_method.png
|       |-- 04_tenure_distribution_by_churn.png
|       |-- 05_monthly_charges_by_churn.png
|       |-- 06_confusion_matrix_logistic_regression.png
|       |-- 07_baseline_metrics_comparison.png
|       `-- manifest.json
|-- src/
|   `-- telco_churn_mlops/
|       |-- api.py                    # FastAPI: /health e /predict
|       |-- baselines.py              # Treino e avaliacao dos baselines
|       |-- data.py                   # Carga, limpeza e split dos dados
|       |-- features.py               # Preprocessing tabular
|       |-- inference.py              # Carregamento e predicao do modelo
|       |-- mlflow_tracking.py         # Tracking opcional com MLflow
|       |-- mlp.py                    # Arquitetura MLP em PyTorch
|       |-- schemas.py                # Schemas Pydantic da API
|       |-- train_baseline_model.py   # Gera artefato usado pela API
|       `-- train_mlp.py              # Treino da MLP
`-- tests/                           # Testes automatizados
```

Observacao: somente a documentacao final da entrega e versionada em `docs/`. Anotacoes internas de sprint e estudo continuam locais e ignoradas pelo Git.

## Requisitos Do Projeto

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

## Resultado Da MLP

Modelo: MLP em PyTorch com duas camadas ocultas, dropout, peso para classe positiva e early stopping.

| Metrica | Valor |
| --- | ---: |
| ROC-AUC | 0,835 |
| F1 | 0,612 |
| Precision | 0,501 |
| Recall | 0,786 |

A MLP ficou muito proxima da Regressao Logistica em ROC-AUC, com F1 e precision ligeiramente maiores, mas recall um pouco menor. Para o objetivo de retencao, a Regressao Logistica continua sendo uma escolha forte para a API por ser simples, interpretavel e ter recall levemente superior.

## Visualizacoes

Os graficos exploratorios sao gerados pelo notebook [01_eda_baselines.ipynb](notebooks/01_eda_baselines.ipynb) e publicados em `reports/figures/`.

- [Distribuicao de churn](reports/figures/01_churn_distribution.png)
- [Churn por tipo de contrato](reports/figures/02_churn_by_contract.png)
- [Churn por metodo de pagamento](reports/figures/03_churn_by_payment_method.png)
- [Tenure por churn](reports/figures/04_tenure_distribution_by_churn.png)
- [MonthlyCharges por churn](reports/figures/05_monthly_charges_by_churn.png)
- [Matriz de confusao](reports/figures/06_confusion_matrix_logistic_regression.png)
- [Comparacao de metricas dos baselines](reports/figures/07_baseline_metrics_comparison.png)

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

Em ambiente Windows CPU, a versao validada do PyTorch foi:

```bash
pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

## Como Reproduzir

Clone o repositorio:

```bash
git clone https://github.com/andreluizpedroso/telco-churn-mlops.git
cd telco-churn-mlops
```

Crie o ambiente virtual e instale as dependencias:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Configure o caminho do pacote no PowerShell:

```bash
$env:PYTHONPATH="src"
```

Baixe o dataset publico Telco Customer Churn:

```bash
python scripts/download_data.py
```

O arquivo sera salvo localmente em:

```text
data/raw/Telco-Customer-Churn.csv
```

Gere os dados processados e as metricas dos baselines:

```bash
python -m telco_churn_mlops.baselines
```

Se MLflow estiver instalado, esse comando tambem registra parametros, metricas e artefatos no experimento `telco-churn-mlops`.

Gere o artefato usado pela API:

```bash
python -m telco_churn_mlops.train_baseline_model
```

Suba a API:

```bash
uvicorn telco_churn_mlops.api:app --reload
```

Acesse:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

Para reproduzir os graficos, abra e execute:

```text
notebooks/01_eda_baselines.ipynb
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

Esse comando gera artefatos locais em `models/` e registra a run no MLflow quando a dependencia esta disponivel.

Abrir a interface local do MLflow:

```bash
mlflow ui
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
- Graficos exploratorios em `reports/figures/`.
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
