# Model Card: Telco Churn Prediction

## Visao Geral

Este Model Card descreve o modelo de classificacao de churn desenvolvido para previsao de cancelamento de clientes em telecomunicacoes.

O objetivo do modelo e estimar a probabilidade de um cliente de telecomunicacoes cancelar o servico, permitindo priorizar acoes de retencao.

## Problema De Negocio

Uma operadora de telecomunicacoes esta perdendo clientes. A empresa precisa identificar clientes com maior risco de churn para agir antes do cancelamento.

## Tipo De Modelo

Modelos considerados:

- `DummyClassifier`: baseline simples que prediz a classe majoritaria.
- Regressao Logistica balanceada: baseline principal usado na API.
- MLP em PyTorch: arquitetura implementada, com treino pendente no ambiente ativo.

## Dataset

Dataset: Telco Customer Churn.

Variavel alvo:

- `Churn`: indica se o cliente cancelou ou nao.

Principais grupos de variaveis:

- Perfil do cliente: genero, senioridade, parceiro, dependentes.
- Servicos contratados: internet, telefone, streaming, suporte.
- Contrato e pagamento: tipo de contrato, cobranca digital, metodo de pagamento.
- Valores: mensalidade e cobranca total.

## Preparacao Dos Dados

Tratamentos aplicados:

- Conversao de `TotalCharges` para numerico.
- Remocao de linhas com `TotalCharges` invalido.
- Conversao de `Churn` para binario: `No = 0`, `Yes = 1`.
- Split estratificado entre treino e teste.
- Encoding de variaveis categoricas com OneHotEncoder.
- Padronizacao de variaveis numericas com StandardScaler.

## Metricas Do Baseline Principal

Modelo: Regressao Logistica balanceada.

Resultados no conjunto de teste:

| Metrica | Valor |
| --- | ---: |
| Accuracy | 0,726 |
| ROC-AUC | 0,835 |
| F1 | 0,607 |
| Precision | 0,490 |
| Recall | 0,797 |

Matriz de confusao:

| | Predito Nao Churn | Predito Churn |
| --- | ---: | ---: |
| Real Nao Churn | 723 | 310 |
| Real Churn | 76 | 298 |

## Interpretacao

O baseline de Regressao Logistica prioriza recall, identificando cerca de 79,7% dos clientes que realmente deram churn no conjunto de teste.

Esse comportamento faz sentido para o problema de negocio, porque deixar de identificar um cliente com alto risco pode custar mais caro do que abordar alguns clientes que nao cancelariam.

## Limitacoes

- O dataset e publico e pode nao representar a distribuicao real de clientes da empresa.
- O modelo nao incorpora historico temporal detalhado.
- O modelo pode perder desempenho se o comportamento dos clientes mudar.
- A MLP ainda precisa ser treinada e comparada no ambiente ativo com PyTorch.
- O registro com MLflow ainda precisa ser executado quando a dependencia estiver disponivel.

## Vieses E Riscos

Possiveis riscos:

- Variaveis de perfil podem capturar padroes sensiveis ou proxies de grupos demograficos.
- Campanhas baseadas no modelo podem tratar clientes de forma desigual.
- Um threshold mal escolhido pode gerar excesso de falsos positivos ou falsos negativos.

Mitigacoes sugeridas:

- Monitorar metricas por segmentos.
- Revisar features usadas em producao.
- Ajustar threshold com base em custo de negocio.
- Ter revisao humana para campanhas sensiveis.

## Uso Recomendado

O modelo deve ser usado para priorizacao de acoes de retencao, nao como decisao automatica final.

Exemplo de uso adequado:

- Gerar uma lista de clientes com maior risco para campanhas de relacionamento.

Exemplo de uso inadequado:

- Negar atendimento, beneficio ou servico com base apenas na previsao.

## Proximas Melhorias

- Treinar e avaliar a MLP.
- Registrar experimentos no MLflow.
- Comparar thresholds usando custo de negocio.
- Adicionar monitoramento por segmento.
- Avaliar drift de dados e de performance.
