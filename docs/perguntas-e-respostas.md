# Perguntas e Respostas

## Sprint 1

### 1. Por que organizar as pastas antes de criar o modelo?

Porque projetos de machine learning geram muitos tipos de arquivos: dados, notebooks, modelos, scripts, testes e documentacao. Se tudo fica misturado, o projeto se torna dificil de entender, manter e avaliar.

Organizar as pastas desde o inicio ajuda qualquer pessoa a saber onde cada coisa deve ficar.

### 2. O que e churn?

Churn e o cancelamento ou abandono de um servico por parte do cliente. Em telecomunicacoes, isso pode significar cancelar internet, telefone, TV ou outro plano contratado.

Prever churn permite que a empresa identifique clientes com maior risco e tente agir antes do cancelamento.

### 3. Por que nao comecar direto pela rede neural?

Porque precisamos de uma referencia. Modelos simples, chamados baselines, mostram o desempenho minimo esperado. Se a rede neural nao melhorar o resultado ou for muito mais dificil de manter, talvez ela nao seja a melhor solucao.

Em projetos reais, o melhor modelo nao e sempre o mais complexo. E o que resolve o problema com bom desempenho, estabilidade e custo aceitavel.

## Conceitos gerais

### O que e uma metrica de negocio?

E uma forma de medir impacto para a empresa, nao apenas para o modelo. Por exemplo: quanto dinheiro pode ser economizado ao evitar churn? Quanto custa oferecer desconto para um cliente que talvez nao cancelasse?

### O que e MLflow?

MLflow e uma ferramenta para registrar experimentos de machine learning. Ela guarda parametros, metricas, modelos e artefatos. Isso facilita comparar diferentes tentativas e reproduzir resultados.

### O que e uma API de inferencia?

E uma aplicacao que recebe dados de entrada e devolve uma previsao do modelo. Neste projeto, a API recebera informacoes de um cliente e retornara o risco de churn.

## Sprint 2

### 1. Por que a acuracia pode enganar em problemas de churn?

Porque a base pode estar desbalanceada. Neste dataset, a maioria dos clientes nao deu churn. Um modelo que sempre responde "nao vai cancelar" acerta muitos casos, mas nao ajuda a encontrar os clientes em risco.

Por isso usamos metricas como recall, precision, F1 e ROC-AUC.

### 2. O que significa validacao estratificada?

Significa separar treino e teste mantendo uma proporcao parecida da variavel alvo em cada parte. Se cerca de 26% dos clientes deram churn na base completa, queremos uma proporcao parecida no treino e no teste.

Isso evita uma avaliacao distorcida por uma separacao de dados desequilibrada.

### 3. Por que converter `TotalCharges` para numero?

Porque modelos de machine learning precisam receber variaveis numericas ou categorias bem codificadas. No dataset original, `TotalCharges` vem como texto e alguns valores estao em branco.

Ao converter para numero, conseguimos detectar os valores invalidos, remover as linhas problematicas e usar essa coluna corretamente no modelo.
