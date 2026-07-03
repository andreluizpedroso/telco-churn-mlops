# Sprint 2: EDA e Baselines

## Objetivo

Entender a base Telco Customer Churn, preparar os dados para modelagem e criar modelos simples de referencia.

## Explicacao didatica

EDA significa analise exploratoria de dados. Nesta etapa, o objetivo e descobrir a qualidade da base antes de treinar modelos: quantidade de linhas, colunas, distribuicao da variavel alvo, tipos de dados e problemas de preenchimento.

Depois da EDA, criamos baselines. Um baseline e um modelo simples que serve como ponto de comparacao. Se um modelo mais complexo nao superar o baseline, ele precisa ser revisto.

## Decisoes iniciais

- Dataset: Telco Customer Churn.
- Variavel alvo: `Churn`.
- Identificador removido das features: `customerID`.
- Correcao aplicada: `TotalCharges` convertido para numerico.
- Linhas com `TotalCharges` vazio sao removidas.
- Alvo convertido para binario: `No = 0`, `Yes = 1`.
- Separacao treino/teste com estratificacao.
- Seed fixa: `42`.

## Modelos baseline

1. `DummyClassifier` com classe majoritaria.
2. Regressao Logistica com `class_weight="balanced"`.

## Resultados iniciais

Depois da limpeza, a base ficou com:

- 7.032 linhas.
- 21 colunas.
- 1.869 clientes com churn.
- 5.163 clientes sem churn.
- Taxa de churn de aproximadamente 26,58%.

Resultados no conjunto de teste:

| Modelo | Accuracy | ROC-AUC | F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| DummyClassifier | 0,734 | 0,500 | 0,000 | 0,000 | 0,000 |
| Regressao Logistica | 0,726 | 0,835 | 0,607 | 0,490 | 0,797 |

Interpretacao: o `DummyClassifier` sempre prediz a classe majoritaria, por isso tem accuracy aparentemente alta, mas nao identifica nenhum cliente com churn. A Regressao Logistica e um baseline muito mais util porque captura cerca de 79,7% dos clientes que realmente deram churn no conjunto de teste.

## Metricas

- Accuracy
- ROC-AUC
- F1-score
- Precision
- Recall
- Matriz de confusao

## Como executar

```bash
python -m telco_churn_mlops.baselines
```

O comando gera:

- `data/processed/train.csv`
- `data/processed/test.csv`
- `docs/baseline-results.json`

## Proximos passos

- Adicionar registro dos experimentos no MLflow quando a dependencia estiver instalada.
- Usar os resultados dos baselines como comparacao para a MLP da Sprint 3.
