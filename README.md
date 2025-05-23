Fazer um part-of-speech tagger (PoS-tagger) e fazer um relatório, inclusive reportando resultados de testes
As três fases do desenvolvimento (no PTB) são:

Treino (usando o corpus de treino; no caso do Penn Treebank usar Seções 0-18);
Teste de desenvolvimento ou validação (usando o corpus de desenvolvimento, validação ou heldout; no caso do Penn Treebank usar Seções 19-21);
Avaliação final (usando o corpus de teste final; no caso do Penn Treebank usar Seções 22-24).
Tipicamente as fases 1 e 2 são repetidas várias vezes quando existem parâmetros que podem ser corrigidos ou ajustados e os resultados dos testes de desenvolvimento podem ser analisados em detalhe comparando com o "gold-standard" ("ground-truth") para melhorar o modelo.

O corpus de teste final não pode ser usado para melhorar o modelo, apenas para reportar resultados do modelo.

O teste básico de uma tarefa de classificação, como é o caso de um part-of-speech tagger é a acurácia, ou taxa de acerto (ou taxa de erro), que consiste simplesmente na divisão do número de tokens classificados corretamente pelo número total de tokens submetido à classificação.

O relatório também deverá apresentar a "confusion matrix" ou matriz de contingência do teste. Esta matriz, na verdade é muito útil para análise dos testes de desenvolvimento, para dar uma visão geral dos casos em que o algoritmo está tendo mais problemas. A análise individual por sentenças, utilizando por exemplo tgrep, é bem mais trabalhosa.

======

Algumas estratégias:

"Baseline": Para cada palavra usar a tag que aparece com maior frequência para aquela palavra no corpus (unigrama).
Experimentar com bigramas e trigramas
Modelar palavra desconhecida através das palavras que aparecem poucas vezes no corpus de treino (1 vez, até 5 vezes, etc.)
Experimentar condicionar com tag(s) da(s) palavra(s) anteriores (processando da esquerda para a direita)
Experimentar com alguns tipos de smoothing (e.g., backof, interpolação, criativo ...)
Nota 1: Não esqueça de considerar tokens para BOS e EOS
Nota 2: A tag dos BOS/EOS não deve ser contabilizada para avaliação
Nota 3: Também é convenção não contabilizada tags de pontuação para avaliação
Nota 4: Pode se usar o "evalb" para avaliação de acurácia de pos-tagging, convertendo-se a saída para algo como
(S (tag1 word1) (tag2 word2) (tag3 word3) ... (tagn wordn) )
Nota 5: Para testar com sentenças de fora do corpus tem-se que assumir o uso de um tokenizador "compatível" para gerar a entrada. (DISCUTIR)
