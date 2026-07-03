# Roteiro STAR Para Video

Tempo alvo: ate 5 minutos.

## S: Situation

Uma operadora de telecomunicacoes esta perdendo clientes em ritmo acelerado. O desafio e criar uma solucao de machine learning para prever quais clientes possuem maior risco de churn.

O problema e importante porque reter clientes costuma ser mais barato do que conquistar novos clientes. Uma previsao de churn permite priorizar campanhas de retencao e atendimento preventivo.

## T: Task

A tarefa do projeto foi construir uma solucao completa, do dado ate a API.

Objetivos tecnicos:

- Analisar um dataset publico de churn.
- Criar baselines com Scikit-Learn.
- Implementar uma MLP com PyTorch.
- Comparar modelos com metricas tecnicas e de negocio.
- Criar uma API de inferencia com FastAPI.
- Adicionar testes, validacao e logging.
- Documentar limitacoes, riscos e monitoramento.

## A: Action

As principais acoes foram:

1. Estruturei o repositorio com `src/`, `tests/`, `data/`, `models/` e `notebooks/`.
2. Fiz a limpeza da base Telco Customer Churn.
3. Converti `TotalCharges` para numerico e `Churn` para alvo binario.
4. Usei split estratificado para preservar a proporcao de churn.
5. Treinei baselines com `DummyClassifier` e Regressao Logistica.
6. Implementei uma MLP em PyTorch com early stopping.
7. Registrei experimentos, metricas e artefatos com MLflow.
8. Criei uma API FastAPI com `/health` e `/predict`.
9. Usei Pydantic para validar os dados de entrada.
10. Adicionei testes automatizados com Pytest.
11. Documentei Model Card, arquitetura e plano de monitoramento.

## R: Result

O baseline de Regressao Logistica apresentou:

- ROC-AUC de 0,835.
- F1 de 0,607.
- Recall de 0,797.

O modelo dummy teve ROC-AUC de 0,500 e recall de 0,000, mostrando que a Regressao Logistica e um baseline muito mais util.

A MLP em PyTorch apresentou ROC-AUC de 0,835, F1 de 0,612 e recall de 0,786. Ela ficou muito proxima da Regressao Logistica, mas o baseline permaneceu como escolha inicial da API por ser mais simples e ter recall levemente superior.

A API foi implementada e testada. Ela consegue receber dados de um cliente e retornar probabilidade de churn usando o baseline salvo localmente.

## Fechamento

O projeto mostra o ciclo completo de machine learning engineering: organizacao, dados, modelo, avaliacao, API, testes e documentacao.

Como proximos passos, eu avaliaria thresholds com custo de negocio, monitoraria drift em producao e faria deploy em nuvem como bonus.
