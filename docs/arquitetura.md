# Arquitetura da Solucao

## Visao Geral

A solucao segue um fluxo simples de machine learning engineering:

```text
Dataset publico
    |
    v
Limpeza e preparacao dos dados
    |
    v
Treino de baselines e MLP
    |
    v
Avaliacao de metricas
    |
    v
Salvamento de artefatos locais
    |
    v
API FastAPI para inferencia
```

## Componentes

### Dados

Pasta local:

- `data/raw/`: dados originais.
- `data/processed/`: dados tratados.

Essas pastas nao sao versionadas no Git, exceto `.gitkeep`.

### Codigo

Pacote principal:

- `src/telco_churn_mlops/`

Modulos:

- `data.py`: carga, limpeza e split.
- `features.py`: preprocessing.
- `baselines.py`: treino e avaliacao dos baselines.
- `mlp.py`: arquitetura da MLP.
- `train_mlp.py`: treino da MLP.
- `train_baseline_model.py`: treino do baseline usado na API.
- `schemas.py`: contratos Pydantic.
- `inference.py`: carregamento do modelo e predicao.
- `api.py`: endpoints FastAPI.

### Modelos

Pasta local:

- `models/`

Artefatos gerados localmente:

- `baseline_logistic_regression.joblib`
- `baseline_logistic_regression_metrics.json`
- `mlp.pt` quando a MLP for treinada.
- `mlp_preprocessor.joblib` quando a MLP for treinada.

Esses arquivos nao sao versionados no Git.

## Fluxo De Treino

1. Baixar dataset.
2. Executar limpeza.
3. Fazer split estratificado.
4. Treinar baseline.
5. Avaliar metricas.
6. Salvar artefato local.

Comando do baseline:

```bash
python -m telco_churn_mlops.train_baseline_model
```

Comando da MLP:

```bash
python -m telco_churn_mlops.train_mlp
```

## Fluxo De Inferencia

1. Cliente envia JSON para `/predict`.
2. Pydantic valida o payload.
3. `inference.py` converte nomes snake_case para nomes originais do dataset.
4. Pipeline carregado calcula probabilidade de churn.
5. API retorna probabilidade, classe prevista, threshold e nome do modelo.

## API

Endpoints:

- `GET /health`
- `POST /predict`

Comando:

```bash
uvicorn src.telco_churn_mlops.api:app --reload
```

## Decisao Batch vs Real-Time

### Real-Time

Usado quando outro sistema precisa consultar um cliente especifico no momento da interacao.

Exemplo:

- Atendente consulta risco de churn durante contato com cliente.

### Batch

Usado para gerar listas periodicas de clientes em risco.

Exemplo:

- Rodar toda madrugada e enviar lista ao time de CRM.

## Recomendacao

Para o primeiro deploy, o modo batch tende a ser mais simples e controlado. A API real-time pode ser mantida para demonstracao e integracao futura.
