# Etiquetador Morfossintático (POS Tagger) para o Penn Treebank

Este projeto implementa um etiquetador morfossintático (Part-of-Speech Tagger) para o corpus Penn Treebank, com várias abordagens e técnicas de suavização (smoothing). O sistema classifica automaticamente cada palavra de um texto com sua classe gramatical correspondente.

## Estrutura do Projeto

```
├── pos_tagger.py           # Implementação principal do POS Tagger
├── run_experiments.py      # Script para executar todos os experimentos
├── analyze_errors.py       # Análise de erros do modelo
├── convert_to_evalb.py     # Utilitário para conversão para formato evalb
├── Secs0-18                # Corpus de treino (Penn Treebank Seções 0-18)
├── Secs19-21               # Corpus de desenvolvimento (Penn Treebank Seções 19-21)
├── Secs22-24               # Corpus de teste final (Penn Treebank Seções 22-24)
├── results/                # Diretório com resultados e análises detalhadas
```

## Modelos Implementados

Este projeto implementa três tipos principais de modelos de etiquetagem:

1. **Unigrama (Baseline)**: Para cada palavra, usa a tag mais frequente observada no corpus de treino.
2. **Bigrama**: Considera a tag da palavra anterior para decidir a tag atual.
3. **Trigrama**: Considera as tags das duas palavras anteriores para decidir a tag atual.

## Métodos de Suavização (Smoothing)

Para lidar com o problema de esparsidade de dados, três abordagens de suavização foram implementadas:

1. **Backoff**: Recorre a um modelo mais simples quando não há dados suficientes.
2. **Interpolação**: Combina as probabilidades de diferentes modelos com pesos.
3. **Sem Suavização (None)**: Modelo sem técnicas de suavização para comparação.

## Tratamento de Palavras Desconhecidas

O sistema lida com palavras desconhecidas modelando-as através das palavras que aparecem com baixa frequência no corpus de treino (palavras raras).

## Resultados

Após avaliação extensiva, os resultados mostram que:

- O **modelo trigrama com interpolação** alcançou a melhor acurácia (94,13%)
- Os modelos do tipo trigrama tiveram o melhor desempenho médio (94,07%)
- O método de suavização por interpolação teve o melhor desempenho médio (93,35%)

As análises detalhadas, incluindo matrizes de confusão e análises de erros, estão disponíveis no diretório `results/`.

## Execução de Experimentos

Para executar todos os experimentos definidos:

```bash
python run_experiments.py
```

## Avaliação

A avaliação é baseada principalmente na acurácia (taxa de acerto), que é calculada como a divisão do número de tokens classificados corretamente pelo número total de tokens. Seguindo convenções padrão:

- Tags de pontuação não são contabilizadas na avaliação
- Tags de BOS/EOS (início/fim de sentença) não são contabilizadas

## Visualizações

O projeto inclui várias visualizações para análise de resultados:

- Matrizes de confusão para cada modelo
- Gráficos comparativos de acurácia entre modelos
- Análises detalhadas de erros mais comuns

## Conclusões

- Modelos de ordem superior (trigramas) fornecem melhores resultados, confirmando a importância do contexto para a tarefa de POS tagging
- A suavização por interpolação provou ser mais eficaz, especialmente para lidar com casos raros
- O tratamento adequado de palavras desconhecidas é crucial para o desempenho em textos reais
