# Plano de Monitoramento

## Objetivo

Monitorar a API e o modelo de churn depois da disponibilizacao, garantindo estabilidade tecnica e utilidade de negocio.

## Metricas Da API

Metricas tecnicas:

- Latencia media por request.
- Latencia p95 e p99.
- Taxa de erro HTTP 4xx.
- Taxa de erro HTTP 5xx.
- Quantidade de requests por minuto.
- Disponibilidade do endpoint `/health`.

Alertas sugeridos:

- Latencia p95 acima de 1 segundo por 10 minutos.
- Erro 5xx acima de 2% por 5 minutos.
- Endpoint `/health` indisponivel.
- Modelo nao carregado em `/health`.

## Metricas Do Modelo

Metricas de performance:

- ROC-AUC.
- F1.
- Precision.
- Recall.
- Matriz de confusao.

Metricas de negocio:

- Quantidade de clientes classificados como risco de churn.
- Taxa de conversao de campanhas de retencao.
- Custo por cliente abordado.
- Estimativa de churn evitado.
- Custo de falso positivo.
- Custo de falso negativo.

## Monitoramento De Dados

Campos para acompanhar:

- Distribuicao de `tenure`.
- Distribuicao de `MonthlyCharges`.
- Distribuicao de `TotalCharges`.
- Frequencia de categorias em `Contract`.
- Frequencia de categorias em `PaymentMethod`.
- Taxa de valores ausentes ou invalidos.

Alertas sugeridos:

- Aumento relevante de valores ausentes.
- Categoria nova em variavel categorica.
- Mudanca brusca na media de mensalidade.
- Mudanca brusca na proporcao de contratos month-to-month.

## Drift

Tipos de drift:

- Data drift: mudanca na distribuicao das features.
- Concept drift: mudanca na relacao entre features e churn.
- Performance drift: queda nas metricas do modelo.

Frequencia sugerida:

- Monitoramento tecnico: continuo.
- Monitoramento de dados: diario ou semanal.
- Monitoramento de performance: mensal, quando houver labels reais de churn.

## Playbook De Resposta

### API Indisponivel

1. Verificar logs da aplicacao.
2. Verificar se o artefato do modelo existe.
3. Reiniciar servico.
4. Se persistir, voltar para uma versao anterior estavel.

### Modelo Com Queda De Performance

1. Confirmar se houve mudanca nos dados de entrada.
2. Avaliar metricas por segmento.
3. Recalcular threshold.
4. Retreinar modelo com dados mais recentes.
5. Comparar novo modelo contra baseline.

### Aumento De Falsos Positivos

1. Revisar threshold.
2. Calcular custo de campanhas desnecessarias.
3. Ajustar criterio de priorizacao.
4. Validar com time de negocio.

### Aumento De Falsos Negativos

1. Revisar recall.
2. Reduzir threshold se o custo de churn perdido for alto.
3. Avaliar novas features.
4. Planejar retreinamento.
