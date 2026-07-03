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
