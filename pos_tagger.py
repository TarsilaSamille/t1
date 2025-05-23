#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Part-of-Speech Tagger para o Penn Treebank.
Implementação seguindo as estratégias especificadas:
- Baseline (unigrama): usando a tag mais frequente para cada palavra
- Abordagens com bigramas e trigramas
- Tratamento de palavras desconhecidas
- Condicionamento com tags de palavras anteriores
- Técnicas de smoothing
"""

import os
import re
import sys
import random
from collections import defaultdict, Counter

# Marcadores de início (Beginning of Sentence) e fim (End of Sentence) de sentença
BOS = "<s>"
EOS = "</s>"

# Tags que representam pontuação (não devem ser contabilizadas para avaliação)
PUNCTUATION_TAGS = set(['.',',','``',"''",':','(',')','-LRB-','-RRB-',';','--','-',])

class POSTagger:
    def __init__(self):
        # Dicionários para armazenar dados estatísticos
        self.word_tag_counts = defaultdict(Counter)  # contagem de tags para cada palavra
        self.tag_counts = Counter()                  # contagem total de cada tag
        self.tag_bigram_counts = defaultdict(Counter)  # contagem de bigramas de tags (tag1, tag2)
        self.tag_trigram_counts = defaultdict(Counter)  # contagem de trigramas de tags (tag1, tag2, tag3)
        self.rare_words = set()                     # conjunto de palavras raras (baixa frequência)
        self.word_counts = Counter()                # contagem total de cada palavra
        self.tags = set()                           # conjunto de todas as tags encontradas

        # Probabilidades para diferentes modelos
        self.unigram_probs = {}                     # P(tag|palavra) - modelo baseline
        self.bigram_probs = defaultdict(dict)       # P(tag|palavra, tag_anterior)
        self.trigram_probs = defaultdict(dict)      # P(tag|palavra, tag_anterior1, tag_anterior2)
        
        # Model settings
        self.rare_word_threshold = 5                # limite para considerar uma palavra como rara
        self.smoothing_method = "backoff"           # método de smoothing: backoff, interpolation, etc.
        
    def preprocess_line(self, line):
        """Processa uma linha do corpus e extrai pares (palavra, tag)."""
        tokens = []
        for token in line.strip().split():
            if '_' in token:
                word, tag = token.rsplit('_', 1)
                tokens.append((word, tag))
                
        return tokens
        
    def train(self, corpus_file, model_type="unigram"):
        """Treina o tagger com um arquivo do corpus."""
        print(f"Treinando modelo {model_type} com o arquivo: {corpus_file}")
        
        # Inicializa contadores
        sentences = []
        word_freq = Counter()
        
        with open(corpus_file, 'r', encoding='utf-8') as file:
            for line in file:
                tokens = self.preprocess_line(line)
                if tokens:
                    # Adiciona marcadores BOS e EOS
                    tokens_with_markers = [(BOS, BOS)] + tokens + [(EOS, EOS)]
                    sentences.append(tokens_with_markers)
                    
                    # Conta frequência das palavras para identificar palavras raras
                    for word, _ in tokens:
                        word_freq[word] += 1
        
        # Identifica palavras raras baseado no threshold
        self.rare_words = {word for word, count in word_freq.items() 
                          if count <= self.rare_word_threshold}
        print(f"Palavras raras identificadas: {len(self.rare_words)}")
        
        # Calcula estatísticas em todas as sentenças
        for sentence in sentences:
            for i in range(len(sentence)):
                word, tag = sentence[i]
                self.word_counts[word] += 1
                
                # Não contabilizamos as tags BOS/EOS nas estatísticas gerais, mas precisamos delas para contexto
                if tag != BOS and tag != EOS:
                    self.word_tag_counts[word][tag] += 1
                    self.tag_counts[tag] += 1
                    self.tags.add(tag)
                
                # Calcula estatísticas para bigramas e trigramas
                if i > 0:  # Para bigramas, precisamos de pelo menos 2 tokens
                    prev_word, prev_tag = sentence[i-1]
                    self.tag_bigram_counts[prev_tag][tag] += 1
                    
                if i > 1:  # Para trigramas, precisamos de pelo menos 3 tokens
                    prev2_word, prev2_tag = sentence[i-2]
                    prev1_word, prev1_tag = sentence[i-1]
                    self.tag_trigram_counts[(prev2_tag, prev1_tag)][tag] += 1
        
        # Calcula probabilidades para o modelo unigrama (baseline)
        self._calculate_unigram_probs()
        
        if model_type in ["bigram", "trigram"]:
            self._calculate_bigram_probs()
            
        if model_type == "trigram":
            self._calculate_trigram_probs()
            
        print(f"Treinamento concluído. Tags identificadas: {len(self.tags)}")
        print(f"Vocabulário de treinamento: {len(self.word_tag_counts)} palavras")
    
    def _calculate_unigram_probs(self):
        """Calcula probabilidades P(tag|palavra) para o modelo unigrama (baseline)."""
        self.unigram_probs = {}
        for word, tag_counter in self.word_tag_counts.items():
            self.unigram_probs[word] = tag_counter.most_common(1)[0][0]  # tag mais frequente
    
    def _calculate_bigram_probs(self):
        """Calcula probabilidades para o modelo bigrama P(tag|palavra, tag_anterior)."""
        # Calculamos P(tag_i | tag_{i-1})
        for prev_tag, tag_counter in self.tag_bigram_counts.items():
            total = sum(tag_counter.values())
            for tag, count in tag_counter.items():
                self.bigram_probs[prev_tag][tag] = count / total if total > 0 else 0
    
    def _calculate_trigram_probs(self):
        """Calcula probabilidades para o modelo trigrama P(tag|palavra, tag_anterior1, tag_anterior2)."""
        for (prev2_tag, prev1_tag), tag_counter in self.tag_trigram_counts.items():
            total = sum(tag_counter.values())
            for tag, count in tag_counter.items():
                self.trigram_probs[(prev2_tag, prev1_tag)][tag] = count / total if total > 0 else 0
    
    def get_most_likely_tag_unigram(self, word):
        """Retorna a tag mais provável para uma palavra no modelo unigrama."""
        if word in self.unigram_probs:
            return self.unigram_probs[word]
        else:
            # Para palavras desconhecidas, usamos heurísticas
            return self._handle_unknown_word(word)
    
    def _handle_unknown_word(self, word):
        """Trata palavras desconhecidas usando heurísticas."""
        # Algumas heurísticas simples baseadas em características da palavra
        if word[0].isupper():
            return 'NNP'  # Nomes próprios geralmente começam com maiúscula
        if word.isdigit():
            return 'CD'   # Números são Cardinal Digits
        if word.endswith('ly'):
            return 'RB'   # Palavras terminadas em 'ly' são geralmente advérbios
        if word.endswith('ing'):
            return 'VBG'  # Gerúndio
        if word.endswith('ed'):
            return 'VBD'  # Verbo no passado
        if word.endswith('s'):
            return 'NNS'  # Substantivo plural
        # Default para substantivos singulares, tag mais comum
        return 'NN'
    
    def tag_sentence_unigram(self, sentence):
        """Taga uma sentença usando o modelo unigrama (baseline)."""
        tagged_tokens = []
        for word in sentence:
            tag = self.get_most_likely_tag_unigram(word)
            tagged_tokens.append((word, tag))
        return tagged_tokens
    
    def tag_sentence_bigram(self, sentence):
        """Taga uma sentença usando o modelo bigrama com smoothing."""
        tagged_tokens = []
        prev_tag = BOS
        
        for word in sentence:
            best_tag = None
            max_prob = -1
            
            # Se a palavra foi vista no treinamento, considere apenas as tags possíveis para ela
            possible_tags = list(self.word_tag_counts.get(word, {}).keys())
            
            # Se a palavra é desconhecida ou não há tags registradas, considere todas as tags possíveis
            if not possible_tags:
                # Para palavras desconhecidas, use heurísticas para reduzir o espaço de busca
                default_tag = self._handle_unknown_word(word)
                possible_tags = [default_tag]  # Começa com a tag mais provável para palavras desconhecidas
                
                # Adiciona outras tags comuns para expandir as possibilidades
                common_tags = ['NN', 'NNP', 'JJ', 'VB']  
                for tag in common_tags:
                    if tag not in possible_tags and tag in self.tags:
                        possible_tags.append(tag)
            
            for tag in possible_tags:
                # Probabilidade do bigrama P(tag | prev_tag)
                bigram_prob = 0
                if prev_tag in self.bigram_probs and tag in self.bigram_probs[prev_tag]:
                    bigram_prob = self.bigram_probs[prev_tag][tag]
                
                # Probabilidade da observação P(word | tag)
                word_given_tag_prob = 0
                if word in self.word_tag_counts and tag in self.word_tag_counts[word]:
                    word_given_tag_prob = self.word_tag_counts[word][tag] / self.tag_counts[tag] if self.tag_counts[tag] > 0 else 0
                else:
                    # Para palavras desconhecidas, atribuímos uma pequena probabilidade
                    # que é maior para a tag sugerida pela heurística
                    word_given_tag_prob = 0.1 if tag == self._handle_unknown_word(word) else 0.01
                
                # Aplicamos técnicas de smoothing
                if self.smoothing_method == "backoff":
                    # Se temos informação do bigrama, usamos ela
                    if bigram_prob > 0:
                        prob = bigram_prob * word_given_tag_prob
                    # Caso contrário, recorremos ao modelo unigrama
                    else:
                        prob = word_given_tag_prob * 0.1  # Fator de desconto para backoff
                
                elif self.smoothing_method == "interpolation":
                    # Interpolação linear entre bigrama e unigrama
                    lambda1 = 0.7  # Peso para o modelo bigrama
                    lambda2 = 0.3  # Peso para o modelo unigrama
                    
                    # Probabilidade combinada
                    prob = (lambda1 * bigram_prob + lambda2 * (self.tag_counts[tag] / sum(self.tag_counts.values()))) * word_given_tag_prob
                
                else:  # Sem smoothing
                    prob = bigram_prob * word_given_tag_prob if bigram_prob > 0 else 0
                
                if prob > max_prob:
                    max_prob = prob
                    best_tag = tag
            
            # Se nenhuma probabilidade for encontrada, use o modelo unigrama como fallback
            if best_tag is None or max_prob == 0:
                best_tag = self.get_most_likely_tag_unigram(word)
            
            tagged_tokens.append((word, best_tag))
            prev_tag = best_tag
            
        return tagged_tokens
    
    def tag_sentence_trigram(self, sentence):
        """Taga uma sentença usando o modelo trigrama com smoothing."""
        tagged_tokens = []
        prev2_tag = BOS
        prev1_tag = BOS
        
        for word in sentence:
            best_tag = None
            max_prob = -1
            
            # Se a palavra foi vista no treinamento, considere apenas as tags possíveis para ela
            possible_tags = list(self.word_tag_counts.get(word, {}).keys())
            
            # Se a palavra é desconhecida ou não há tags registradas, considere tags selecionadas
            if not possible_tags:
                # Para palavras desconhecidas, use heurísticas para reduzir o espaço de busca
                default_tag = self._handle_unknown_word(word)
                possible_tags = [default_tag]  # Começa com a tag mais provável para palavras desconhecidas
                
                # Adiciona outras tags comuns para expandir as possibilidades
                common_tags = ['NN', 'NNP', 'JJ', 'VB']  
                for tag in common_tags:
                    if tag not in possible_tags and tag in self.tags:
                        possible_tags.append(tag)
            
            for tag in possible_tags:
                # Probabilidade do trigrama P(tag | prev2_tag, prev1_tag)
                trigram_prob = 0
                if (prev2_tag, prev1_tag) in self.trigram_probs and tag in self.trigram_probs[(prev2_tag, prev1_tag)]:
                    trigram_prob = self.trigram_probs[(prev2_tag, prev1_tag)][tag]
                
                # Probabilidade do bigrama P(tag | prev1_tag) como backoff
                bigram_prob = 0
                if prev1_tag in self.bigram_probs and tag in self.bigram_probs[prev1_tag]:
                    bigram_prob = self.bigram_probs[prev1_tag][tag]
                
                # Probabilidade da observação P(word | tag)
                word_given_tag_prob = 0
                if word in self.word_tag_counts and tag in self.word_tag_counts[word]:
                    word_given_tag_prob = self.word_tag_counts[word][tag] / self.tag_counts[tag] if self.tag_counts[tag] > 0 else 0
                else:
                    # Para palavras desconhecidas, atribuímos uma pequena probabilidade
                    # que é maior para a tag sugerida pela heurística
                    word_given_tag_prob = 0.1 if tag == self._handle_unknown_word(word) else 0.01
                
                # Aplicamos técnicas de smoothing
                if self.smoothing_method == "backoff":
                    # Tentamos primeiro o modelo trigrama
                    if trigram_prob > 0:
                        prob = trigram_prob * word_given_tag_prob
                    # Backoff para o modelo bigrama
                    elif bigram_prob > 0:
                        prob = bigram_prob * word_given_tag_prob * 0.4  # Fator de desconto para backoff
                    # Backoff para o modelo unigrama (prior da tag)
                    else:
                        prior_prob = self.tag_counts[tag] / sum(self.tag_counts.values())
                        prob = prior_prob * word_given_tag_prob * 0.2  # Fator de desconto maior
                
                elif self.smoothing_method == "interpolation":
                    # Interpolação linear entre trigrama, bigrama e unigrama
                    lambda1 = 0.5  # Peso para o modelo trigrama
                    lambda2 = 0.3  # Peso para o modelo bigrama
                    lambda3 = 0.2  # Peso para o modelo unigrama (prior)
                    
                    # Probabilidade prior da tag
                    prior_prob = self.tag_counts[tag] / sum(self.tag_counts.values())
                    
                    # Probabilidade combinada
                    comb_prob = (lambda1 * trigram_prob + lambda2 * bigram_prob + lambda3 * prior_prob)
                    prob = comb_prob * word_given_tag_prob
                
                else:  # Sem smoothing
                    prob = trigram_prob * word_given_tag_prob if trigram_prob > 0 else 0
                
                if prob > max_prob:
                    max_prob = prob
                    best_tag = tag
            
            # Se nenhuma probabilidade for encontrada, use o modelo unigrama como fallback
            if best_tag is None or max_prob == 0:
                best_tag = self.get_most_likely_tag_unigram(word)
            
            tagged_tokens.append((word, best_tag))
            prev2_tag = prev1_tag
            prev1_tag = best_tag
            
        return tagged_tokens
    
    def tag_sentence(self, sentence, model_type="unigram"):
        """Taga uma sentença de acordo com o modelo especificado."""
        # Adiciona marcadores de início e fim
        words = sentence.strip().split()
        
        if model_type == "unigram":
            return self.tag_sentence_unigram(words)
        elif model_type == "bigram":
            return self.tag_sentence_bigram(words)
        elif model_type == "trigram":
            return self.tag_sentence_trigram(words)
        else:
            raise ValueError(f"Modelo desconhecido: {model_type}")
    
    def evaluate(self, test_file, model_type="unigram"):
        """Avalia o desempenho do tagger em um arquivo de teste."""
        print(f"Avaliando modelo {model_type} no arquivo: {test_file}")
        
        total_words = 0
        correct_tags = 0
        confusion_matrix = defaultdict(Counter)
        
        with open(test_file, 'r', encoding='utf-8') as file:
            for line in file:
                tokens = self.preprocess_line(line)
                if not tokens:
                    continue
                
                # Separa palavras e tags gold
                words = [word for word, _ in tokens]
                gold_tags = [tag for _, tag in tokens]
                
                # Prediz tags com o modelo
                tagged_tokens = self.tag_sentence(" ".join(words), model_type)
                predicted_tags = [tag for _, tag in tagged_tokens]
                
                # Avalia os resultados (ignorando BOS e EOS, e pontuação)
                for i in range(len(gold_tags)):
                    if i < len(predicted_tags):  # Verifica se o índice é válido
                        # Ignora pontuação na avaliação
                        if gold_tags[i] not in PUNCTUATION_TAGS:
                            total_words += 1
                            if gold_tags[i] == predicted_tags[i]:
                                correct_tags += 1
                            
                            # Atualiza matriz de confusão
                            confusion_matrix[gold_tags[i]][predicted_tags[i]] += 1
        
        # Calcula acurácia
        accuracy = correct_tags / total_words if total_words > 0 else 0
        print(f"Acurácia do modelo {model_type}: {accuracy:.4f} ({correct_tags}/{total_words})")
        
        # Retorna acurácia e matriz de confusão
        return accuracy, confusion_matrix
    
    def print_confusion_matrix(self, confusion_matrix, top_n=10):
        """Imprime as entradas mais significativas da matriz de confusão."""
        print("\nMatriz de Confusão (Top {}):".format(top_n))
        
        # Encontra todas as tags presentes na matriz
        all_tags = set()
        for gold_tag, pred_counts in confusion_matrix.items():
            all_tags.add(gold_tag)
            for pred_tag in pred_counts:
                all_tags.add(pred_tag)
        
        # Cria uma lista de pares (gold_tag, pred_tag, count) para todos os erros
        confusions = []
        for gold_tag, pred_counts in confusion_matrix.items():
            for pred_tag, count in pred_counts.items():
                if gold_tag != pred_tag:  # Apenas erros
                    confusions.append((gold_tag, pred_tag, count))
        
        # Ordena por contagem e imprime os top_n
        confusions.sort(key=lambda x: x[2], reverse=True)
        print(f"{'Gold Tag':<10} {'Pred Tag':<10} {'Count':<10}")
        print("-" * 30)
        for gold_tag, pred_tag, count in confusions[:top_n]:
            print(f"{gold_tag:<10} {pred_tag:<10} {count:<10}")
    
    def convert_to_evalb_format(self, sentence, tags):
        """Converte uma sentença taggeada para o formato compatível com evalb."""
        result = "(S"
        for word, tag in zip(sentence.split(), tags):
            result += f" ({tag} {word})"
        result += " )"
        return result
    
    def save_model(self, filename):
        """Salva o modelo treinado em um arquivo."""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump({
                'word_tag_counts': dict(self.word_tag_counts),
                'tag_counts': self.tag_counts,
                'tag_bigram_counts': dict(self.tag_bigram_counts),
                'tag_trigram_counts': dict(self.tag_trigram_counts),
                'unigram_probs': self.unigram_probs,
                'bigram_probs': dict(self.bigram_probs),
                'trigram_probs': dict(self.trigram_probs),
                'rare_words': self.rare_words,
                'word_counts': self.word_counts,
                'tags': self.tags,
                'rare_word_threshold': self.rare_word_threshold,
                'smoothing_method': self.smoothing_method
            }, f)
        print(f"Modelo salvo em {filename}")
    
    def load_model(self, filename):
        """Carrega um modelo treinado a partir de um arquivo."""
        import pickle
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.word_tag_counts = defaultdict(Counter, data['word_tag_counts'])
            self.tag_counts = data['tag_counts']
            self.tag_bigram_counts = defaultdict(Counter, data['tag_bigram_counts'])
            self.tag_trigram_counts = defaultdict(Counter, data['tag_trigram_counts'])
            self.unigram_probs = data['unigram_probs']
            self.bigram_probs = defaultdict(dict, data['bigram_probs'])
            self.trigram_probs = defaultdict(dict, data['trigram_probs'])
            self.rare_words = data['rare_words']
            self.word_counts = data['word_counts']
            self.tags = data['tags']
            self.rare_word_threshold = data['rare_word_threshold']
            self.smoothing_method = data['smoothing_method']
        print(f"Modelo carregado de {filename}")

def main():
    # Verifica argumentos da linha de comando
    if len(sys.argv) < 3:
        print("Uso: python pos_tagger.py <modo> <arquivo_entrada> [modelo] [smoothing]")
        print("Modos disponíveis: train, tag, eval")
        print("Modelos disponíveis: unigram (default), bigram, trigram")
        print("Métodos de smoothing: backoff (default), interpolation, none")
        sys.exit(1)
    
    mode = sys.argv[1]
    input_file = sys.argv[2]
    model_type = sys.argv[3] if len(sys.argv) > 3 else "unigram"
    smoothing = sys.argv[4] if len(sys.argv) > 4 else "backoff"
    
    tagger = POSTagger()
    tagger.smoothing_method = smoothing
    
    if mode == "train":
        tagger.train(input_file, model_type)
        # Salva o modelo treinado
        model_file = f"tagger_{model_type}_{smoothing}.pkl"
        tagger.save_model(model_file)
    
    elif mode == "tag":
        # Carrega o modelo se disponível
        model_file = f"tagger_{model_type}_{smoothing}.pkl"
        if os.path.exists(model_file):
            tagger.load_model(model_file)
        else:
            print(f"Modelo {model_file} não encontrado. Treine primeiro.")
            sys.exit(1)
        
        # Processa linha por linha do arquivo de entrada
        with open(input_file, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    # Para arquivos sem tags, assume-se que são apenas palavras
                    if '_' not in line:
                        tagged_tokens = tagger.tag_sentence(line, model_type)
                        # Imprime o resultado no formato word_TAG
                        print(" ".join(f"{word}_{tag}" for word, tag in tagged_tokens))
                    # Para arquivos com tags, usa as palavras e compara com as tags reais
                    else:
                        tokens = tagger.preprocess_line(line)
                        words = [word for word, _ in tokens]
                        gold_tags = [tag for _, tag in tokens]
                        
                        tagged_tokens = tagger.tag_sentence(" ".join(words), model_type)
                        pred_tags = [tag for _, tag in tagged_tokens]
                        
                        # Imprime lado a lado para comparação
                        print("GOLD: " + " ".join(f"{word}_{tag}" for word, tag in zip(words, gold_tags)))
                        print("PRED: " + " ".join(f"{word}_{tag}" for word, tag in zip(words, pred_tags)))
                        print("")
    
    elif mode == "eval":
        # Carrega o modelo se disponível
        model_file = f"tagger_{model_type}_{smoothing}.pkl"
        if os.path.exists(model_file):
            tagger.load_model(model_file)
        else:
            print(f"Modelo {model_file} não encontrado. Treine primeiro.")
            sys.exit(1)
        
        # Avalia o modelo no arquivo de teste
        accuracy, confusion_matrix = tagger.evaluate(input_file, model_type)
        
        # Imprime os resultados detalhados
        print(f"Acurácia final: {accuracy:.4f}")
        tagger.print_confusion_matrix(confusion_matrix)
        
        # Salva os resultados em um arquivo
        result_file = f"results_{model_type}_{smoothing}.txt"
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"Modelo: {model_type}, Smoothing: {smoothing}\n")
            f.write(f"Acurácia: {accuracy:.4f}\n\n")
            
            f.write("Matriz de Confusão (Top 20):\n")
            f.write(f"{'Gold Tag':<10} {'Pred Tag':<10} {'Count':<10}\n")
            f.write("-" * 30 + "\n")
            
            confusions = []
            for gold_tag, pred_counts in confusion_matrix.items():
                for pred_tag, count in pred_counts.items():
                    if gold_tag != pred_tag:  # Apenas erros
                        confusions.append((gold_tag, pred_tag, count))
            
            confusions.sort(key=lambda x: x[2], reverse=True)
            for gold_tag, pred_tag, count in confusions[:20]:
                f.write(f"{gold_tag:<10} {pred_tag:<10} {count:<10}\n")
        
        print(f"Resultados detalhados salvos em {result_file}")
    
    else:
        print(f"Modo desconhecido: {mode}")
        print("Modos disponíveis: train, tag, eval")
        sys.exit(1)

if __name__ == "__main__":
    main()
