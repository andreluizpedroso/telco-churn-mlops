# Telco Churn MLOps

Projeto do Tech Challenge Fase 01 para prever churn de clientes de telecomunicacoes usando boas praticas de Machine Learning Engineering.

## Problema

Churn acontece quando um cliente cancela ou deixa de usar um servico. Em uma operadora de telecomunicacoes, prever churn ajuda o time de negocio a agir antes do cancelamento, oferecendo atendimento, beneficios ou campanhas de retencao para clientes com maior risco.

Neste projeto, o objetivo e criar um modelo preditivo que classifique clientes com risco de churn e disponibilizar esse modelo por uma API.

## Como um junior deve pensar neste projeto

Um projeto de machine learning profissional nao comeca pelo modelo. Primeiro precisamos entender o problema, organizar os dados, definir metricas, criar uma referencia simples e so depois comparar modelos mais complexos.

A rede neural MLP sera importante porque faz parte do desafio, mas ela precisa ser comparada com baselines. Se um modelo simples resolver melhor, precisamos saber explicar isso.

## Estrutura do repositorio

```text
.
├── data/
│   ├── raw/          # Dados originais, sem alteracao
│   └── processed/    # Dados tratados para treino e avaliacao
├── docs/             # Documentacao do projeto, sprints e estudos
├── models/           # Modelos e artefatos treinados
├── notebooks/        # Analises exploratorias e experimentos
├── src/              # Codigo reutilizavel do projeto
└── tests/            # Testes automatizados
```

## Sprints

O projeto sera construido em sprints para facilitar aprendizado e entrega incremental.

1. Fundacao do projeto
2. EDA e modelos baseline
3. Rede neural MLP com PyTorch
4. Pipeline, testes e API
5. Documentacao final, Model Card e roteiro STAR

Veja o detalhe em [docs/sprints.md](docs/sprints.md).

## Dataset

A sugestao inicial e usar um dataset publico de churn em telecomunicacoes, como o Telco Customer Churn. O dataset final sera documentado na Sprint 2, junto com as decisoes de tratamento dos dados.

## Comandos planejados

Os comandos abaixo serao usados ao longo do projeto, depois que as dependencias forem instaladas.

```bash
pytest
ruff check .
uvicorn src.telco_churn_mlops.api:app --reload
```

## Entregas esperadas

- Codigo organizado e versionado no GitHub.
- Analise exploratoria dos dados.
- Baselines com Scikit-Learn.
- MLP treinada com PyTorch.
- Experimentos rastreados com MLflow.
- API de inferencia com FastAPI.
- Testes automatizados.
- Model Card e plano de monitoramento.
- Video final de ate 5 minutos usando o metodo STAR.
