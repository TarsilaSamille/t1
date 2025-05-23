#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para gerar matrizes de confusão melhoradas para visualização dos resultados
do POS Tagger, destacando a diagonal principal e facilitando a análise dos erros.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import re

# Diretório onde os resultados estão salvos
RESULTS_DIR = "results"

def extract_confusion_matrix(filename):
    """Extrai a matriz de confusão de um arquivo de resultados."""
    confusion_matrix = defaultdict(Counter)
    
    # Extrai acurácia geral para armazenar no dicionário
    accuracy = None
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # Extrai a acurácia geral
        accuracy_pattern = r"Acurácia.*?(\d+\.\d+)"
        accuracy_match = re.search(accuracy_pattern, content)
        if accuracy_match:
            accuracy = float(accuracy_match.group(1))
        
        # Procura pelo bloco da matriz de confusão
        confusion_pattern = r"Matriz de Confusão.*?\n(.*?)(?:\n\n|\Z)"
        match = re.search(confusion_pattern, content, re.DOTALL)
        
        if match:
            confusion_block = match.group(1)
            lines = confusion_block.strip().split('\n')
            
            # Pula o cabeçalho e a linha de separação
            data_lines = []
            for line in lines:
                if "-----" in line:  # Linha de separação
                    continue
                if "Gold Tag" in line:  # Cabeçalho
                    continue
                if "Resultados detalhados salvos" in line:
                    continue
                if line.strip():
                    data_lines.append(line)
            
            for line in data_lines:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        gold_tag = parts[0]
                        pred_tag = parts[1]
                        count = int(parts[2])
                        confusion_matrix[gold_tag][pred_tag] = count
                    except ValueError:
                        # Ignora linhas que não seguem o formato esperado
                        continue
        
        # Processa todo o conteúdo para obter os totais corretos
        all_results = re.findall(r"(\S+)\s+(\S+)\s+(\d+)", content)
        tag_counts = defaultdict(int)
        
        # Processa todos os encontrados para contabilizar corretamente os totais
        for gold, pred, count_str in all_results:
            try:
                count = int(count_str)
                tag_counts[gold] += count  # Acumula contagem para cada tag ouro
            except ValueError:
                continue
        
        # Armazena os totais por tag como informação adicional
        for tag, count in tag_counts.items():
            if count > 0:  # Só armazena se houver alguma ocorrência
                confusion_matrix['_total_'][tag] = count
    
    # Armazena a acurácia geral se disponível
    if accuracy is not None:
        confusion_matrix['accuracy'] = accuracy
    
    return confusion_matrix

def plot_improved_confusion_matrix(confusion_matrix, model_name, save_path):
    """Gera uma visualização melhorada da matriz de confusão com a diagonal destacada."""
    # Encontra as tags mais frequentes para reduzir o tamanho da matriz
    tag_counts = defaultdict(int)
    for gold_tag in confusion_matrix:
        if gold_tag in ['accuracy', '_total_']:
            continue
        # Conta a frequência total de cada tag
        for pred_tag, count in confusion_matrix[gold_tag].items():
            tag_counts[gold_tag] += count
            tag_counts[pred_tag] += count
    
    # Seleciona as top N tags mais frequentes (ajuste conforme necessário)
    top_n = 20  # Reduzido para mostrar só as 20 tags mais relevantes
    top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_tags = [tag for tag, count in top_tags]
    
    # Ignorar tags de pontuação que geralmente não são úteis para análise
    punctuation_tags = ['.', ',', ':', ';', "''", '``', '-LRB-', '-RRB-', '--']
    top_tags = [tag for tag in top_tags if tag not in punctuation_tags]
    
    # Garantir que temos no máximo top_n tags
    top_tags = top_tags[:top_n]
    
    # Ordena as tags para manter consistência
    all_tags = sorted(top_tags)
    
    # Cria uma matriz NumPy para os dados
    matrix = np.zeros((len(all_tags), len(all_tags)))
    
    # Preenche a matriz
    for i, gold_tag in enumerate(all_tags):
        for j, pred_tag in enumerate(all_tags):
            if gold_tag in confusion_matrix and pred_tag in confusion_matrix[gold_tag]:
                matrix[i, j] = confusion_matrix[gold_tag][pred_tag]
    
    # Normaliza por linha (ouro)
    row_sums = matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Evita divisão por zero
    matrix_norm = matrix / row_sums[:, np.newaxis]
    
    # Filtra valores muito pequenos (confusões raras) para melhorar a visualização
    threshold = 0.05  # Valores menores que 5% são ignorados visualmente
    matrix_display = matrix_norm.copy()
    matrix_display[matrix_display < threshold] = np.nan  # Não mostrar valores pequenos
    
    # Cria o plot
    plt.figure(figsize=(16, 14))  # Tamanho ajustado para melhor visualização
    
    # Usar uma paleta de cores que destaque bem a diagonal
    # cmap personalizado para melhorar o contraste
    ax = sns.heatmap(matrix_norm, annot=matrix_display, fmt=".2f", cmap="YlOrRd",
                    xticklabels=all_tags, yticklabels=all_tags, 
                    cbar_kws={'label': 'Proporção de Predições'},
                    annot_kws={"size": 10})  # Aumentar tamanho das anotações
    
    # Destaca a diagonal principal
    for i in range(len(all_tags)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='black', lw=2))
    
    # Título mais informativo
    ax.set_title(f'Matriz de Confusão Simplificada - {model_name}\n(Apenas tags mais significativas)', fontsize=16)
    ax.set_xlabel('Tag Prevista', fontsize=14)
    ax.set_ylabel('Tag Correta (Gold)', fontsize=14)
    
    # Rotaciona os labels do eixo x para melhor visibilidade
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    
    # Adiciona informações sobre a acurácia
    accuracy = confusion_matrix.get('accuracy', None)
    if accuracy is not None:
        plt.figtext(0.5, 0.01, f'Acurácia: {accuracy:.4f}', ha='center', fontsize=12)
    
    # Salva o gráfico
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    return matrix, all_tags

def main():
    """Função principal que processa os arquivos de resultado e gera novas matrizes de confusão."""
    # Verifica argumentos ou usa valores padrão
    if len(sys.argv) > 1:
        input_files = sys.argv[1:]
    else:
        # Procura por todos os arquivos de resultado no diretório atual
        input_files = [f for f in os.listdir() if f.startswith('results_') and f.endswith('.txt')]
    
    print(f"Processando {len(input_files)} arquivos: {input_files}")
    
    # Cria diretório de resultados se não existir
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    for filename in input_files:
        # Extrai informações do nome do arquivo
        parts = filename.replace('results_', '').replace('.txt', '').split('_')
        if len(parts) >= 2:
            model_type = parts[0]  # unigram, bigram, trigram
            smoothing = parts[1]   # none, backoff, interpolation
            model_name = f"{model_type.capitalize()} com {smoothing}"
        else:
            model_name = filename.replace('results_', '').replace('.txt', '')
        
        print(f"Processando {filename} ({model_name})...")
        
        # Extrai a matriz de confusão
        confusion_matrix = extract_confusion_matrix(filename)
        
        if not confusion_matrix:
            print(f"Não foi possível extrair a matriz de confusão de {filename}")
            continue
        
        # Gera o plot melhorado
        save_path = os.path.join(RESULTS_DIR, f"matrix_simplificada_{model_type}_{smoothing}.png")
        matrix, all_tags = plot_improved_confusion_matrix(confusion_matrix, model_name, save_path)
        
        print(f"Matriz de confusão simplificada salva em {save_path}")

if __name__ == "__main__":
    main()
