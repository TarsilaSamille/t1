#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para analisar erros comuns e gerar matrizes de confusão visuais
para o modelo POS Tagger.
"""

import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter

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
        
        # Também buscamos outros dados do conteúdo para montar estatísticas mais corretas
        # Tentamos extrair totais por tag para calcular melhor as taxas de erro
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

def plot_simplified_confusion_matrix(confusion_matrix, model_name, save_path):
    """Gera uma visualização simplificada da matriz de confusão com a diagonal destacada."""
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

def analyze_common_errors(confusion_matrix, all_tags, model_name, save_path):
    """Analisa os erros mais comuns e gera um relatório."""
    # Cria um DataFrame para facilitar a análise
    errors = []
    
    # Extrai a acurácia do modelo se disponível
    accuracy = confusion_matrix.get('accuracy', None)
    
    # Obtém os totais por tag, se disponíveis
    tag_totals = dict(confusion_matrix.get('_total_', {}))
    
    for gold_tag in confusion_matrix:
        # Pula as entradas especiais
        if gold_tag in ['_total_', 'accuracy']:
            continue
            
        # Calcula o total de ocorrências desta tag
        # Primeiro tenta usar o total previamente calculado
        gold_total = tag_totals.get(gold_tag, 0)
        
        # Se não tivermos este valor, somamos todas as ocorrências desta tag nos resultados
        if gold_total == 0:
            gold_total = sum(confusion_matrix[gold_tag].values())
            
        for pred_tag in confusion_matrix[gold_tag]:
            if gold_tag != pred_tag:  # Apenas erros
                count = confusion_matrix[gold_tag][pred_tag]
                if count > 0:
                    errors.append({
                        'Gold Tag': gold_tag,
                        'Pred Tag': pred_tag,
                        'Count': count,
                        'Total': gold_total
                    })
    
    # Ordena os erros por contagem
    error_df = pd.DataFrame(errors)
    if not error_df.empty:
        error_df = error_df.sort_values('Count', ascending=False).reset_index(drop=True)
        
        # Gera um relatório de erros
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"ANÁLISE DE ERROS COMUNS - {model_name}\n")
            f.write("=" * 60 + "\n\n")
            
            # Inclui a acurácia geral se disponível
            if accuracy is not None:
                f.write(f"Acurácia geral: {accuracy:.4f}\n\n")
            
            f.write("Top 20 Erros Mais Frequentes:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Gold Tag':<10} {'Pred Tag':<10} {'Count':<10}\n")
            f.write("-" * 60 + "\n")
            
            for _, row in error_df.head(20).iterrows():
                f.write(f"{row['Gold Tag']:<10} {row['Pred Tag']:<10} {row['Count']:<10}\n")
            
            # Analisa erros por tipo de tag
            f.write("\n\nErros por Categoria de Tag:\n")
            f.write("-" * 60 + "\n")
            
            # Agrupa erros por tag de ouro
            tag_errors = defaultdict(int)
            
            for gold_tag in confusion_matrix:
                if gold_tag in ['_total_', 'accuracy']:
                    continue
                    
                # Obtém o total para esta tag ouro
                gold_total = tag_totals.get(gold_tag, 0)
                if gold_total == 0:
                    gold_total = sum(confusion_matrix[gold_tag].values())
                
                # Soma todos os erros para esta tag
                error_sum = sum(count for pred_tag, count in confusion_matrix[gold_tag].items() 
                                if pred_tag != gold_tag)
                
                # Armazena apenas se houver ocorrências suficientes
                if gold_total > 0:
                    tag_errors[gold_tag] = (error_sum, gold_total)
            
            # Calcula a taxa de erro por tag
            error_rates = []
            for tag, (errors, total) in tag_errors.items():
                error_rate = errors / total if total > 0 else 0
                error_rates.append({
                    'Tag': tag,
                    'Total': total,
                    'Errors': errors,
                    'Error Rate': error_rate
                })
            
            # Ordena por taxa de erro
            error_rate_df = pd.DataFrame(error_rates)
            if not error_rate_df.empty:
                error_rate_df = error_rate_df.sort_values('Error Rate', ascending=False).reset_index(drop=True)
                
                f.write(f"{'Tag':<10} {'Total':<10} {'Errors':<10} {'Error Rate':<10}\n")
                f.write("-" * 60 + "\n")
                
                for _, row in error_rate_df.iterrows():
                    f.write(f"{row['Tag']:<10} {row['Total']:<10} {row['Errors']:<10} {row['Error Rate']:.4f}\n")
            
            # Análise de pares específicos de tags confundidas
            f.write("\n\nPares de Tags Frequentemente Confundidos:\n")
            f.write("-" * 60 + "\n")
            
            # Identifica pares de tags que são frequentemente confundidos
            tag_pairs = []
            for gold_tag in confusion_matrix:
                if gold_tag in ['_total_', 'accuracy']:
                    continue
                    
                for pred_tag in confusion_matrix[gold_tag]:
                    if gold_tag != pred_tag:
                        count_gold_pred = confusion_matrix[gold_tag][pred_tag]
                        # Verifica se o par inverso existe
                        count_pred_gold = 0
                        if pred_tag in confusion_matrix and gold_tag in confusion_matrix[pred_tag]:
                            count_pred_gold = confusion_matrix[pred_tag][gold_tag]
                        
                        if count_gold_pred > 0 or count_pred_gold > 0:
                            tag_pairs.append({
                                'Tag1': gold_tag,
                                'Tag2': pred_tag,
                                'Count1to2': count_gold_pred,
                                'Count2to1': count_pred_gold,
                                'Total': count_gold_pred + count_pred_gold
                            })
            
            # Remove duplicatas (ambas as direções)
            unique_pairs = {}
            for pair in tag_pairs:
                key = tuple(sorted([pair['Tag1'], pair['Tag2']]))
                if key not in unique_pairs or unique_pairs[key]['Total'] < pair['Total']:
                    unique_pairs[key] = pair
            
            # Ordena os pares por contagem total
            pairs_df = pd.DataFrame(list(unique_pairs.values()))
            if not pairs_df.empty:
                pairs_df = pairs_df.sort_values('Total', ascending=False).reset_index(drop=True)
                
                f.write(f"{'Tag1':<10} {'Tag2':<10} {'Count1to2':<10} {'Count2to1':<10} {'Total':<10}\n")
                f.write("-" * 60 + "\n")
                
                for _, row in pairs_df.head(20).iterrows():
                    f.write(f"{row['Tag1']:<10} {row['Tag2']:<10} {row['Count1to2']:<10} "
                           f"{row['Count2to1']:<10} {row['Total']:<10}\n")
            
            # Recomendações para melhorias
            f.write("\n\nRECOMENDAÇÕES PARA MELHORIAS:\n")
            f.write("-" * 60 + "\n")
            
            # Identifica as tags com maiores taxas de erro
            if not error_rate_df.empty:
                problem_tags = error_rate_df[error_rate_df['Error Rate'] > 0.2].head(5)
                
                if not problem_tags.empty:
                    f.write("1. Foco nas tags com maiores taxas de erro:\n")
                    for _, row in problem_tags.iterrows():
                        f.write(f"   - {row['Tag']}: Taxa de erro de {row['Error Rate']:.4f}\n")
                
                # Identifica os pares de tags mais confundidos
                if not pairs_df.empty:
                    f.write("\n2. Melhorar a distinção entre pares frequentemente confundidos:\n")
                    for _, row in pairs_df.head(5).iterrows():
                        f.write(f"   - {row['Tag1']} e {row['Tag2']}: {row['Total']} confusões\n")
                
                # Recomendações gerais
                f.write("\n3. Recomendações gerais:\n")
                f.write("   - Considerar features adicionais para melhorar a distinção de tags problemáticas\n")
                f.write("   - Usar técnicas de smoothing mais avançadas\n")
                f.write("   - Experimentar com janelas de contexto maiores\n")
                f.write("   - Melhorar o tratamento de palavras desconhecidas\n")
    
    return error_df if not error_df.empty else None

def analyze_all_models():
    """Analisa todos os modelos e gera relatórios comparativos."""
    if not os.path.exists(RESULTS_DIR):
        print(f"Diretório {RESULTS_DIR} não encontrado.")
        return
    
    # Primeiro, carrega os dados do arquivo de relatório principal
    model_accuracies = {}
    try:
        report_file = os.path.join(RESULTS_DIR, "report.txt")
        if os.path.exists(report_file):
            with open(report_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Extrai informações de acurácia do conjunto de teste
                test_pattern = r"Conjunto de Teste:.*?Modelo\s+Smoothing\s+Acurácia\s+\n[-]+\n(.*?)(?:\n\n|\Z)"
                test_match = re.search(test_pattern, content, re.DOTALL)
                
                if test_match:
                    test_block = test_match.group(1)
                    for line in test_block.strip().split('\n'):
                        parts = line.split()
                        if len(parts) >= 3:
                            model_type = parts[0]
                            smoothing = parts[1]
                            accuracy = float(parts[2])
                            
                            model_name = f"{model_type}_{smoothing}"
                            model_accuracies[model_name] = accuracy
    except Exception as e:
        print(f"Erro ao processar o relatório principal: {e}")
    
    # Procura por todos os arquivos de resultados detalhados
    result_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("detailed_") and f.endswith(".txt")]
    
    if not result_files:
        print("Nenhum arquivo de resultados detalhados encontrado.")
        # Se não há arquivos detalhados, mas temos acurácias, gere apenas o relatório comparativo
        if model_accuracies:
            generate_comparative_report_from_accuracies(model_accuracies)
        return
    
    for result_file in result_files:
        # Extrai nome do modelo da nomenclatura do arquivo
        model_name = result_file.replace("detailed_", "").replace(".txt", "")
        
        print(f"Analisando modelo: {model_name}")
        
        # Caminho completo para o arquivo
        file_path = os.path.join(RESULTS_DIR, result_file)
        
        # Extrai a matriz de confusão
        confusion_matrix = extract_confusion_matrix(file_path)
        
        if not confusion_matrix:
            print(f"Não foi possível extrair a matriz de confusão para {model_name}")
            continue
        
        # Gera apenas a matriz de confusão simplificada
        simplified_path = os.path.join(RESULTS_DIR, f"matrix_simplificada_{model_name}.png")
        simplified_matrix, simplified_tags = plot_simplified_confusion_matrix(confusion_matrix, model_name, simplified_path)
        
        # Analisa os erros comuns
        error_path = os.path.join(RESULTS_DIR, f"error_analysis_{model_name}.txt")
        
        # Adiciona a acurácia do relatório principal para garantir consistência
        if model_name in model_accuracies:
            # Injeta a acurácia no modelo de confusão para análise
            if 'accuracy' not in confusion_matrix:
                confusion_matrix['accuracy'] = model_accuracies[model_name]
        
        # Obtém todas as tags únicas para análise
        all_tags = set()
        for gold_tag in confusion_matrix:
            if gold_tag in ['accuracy', '_total_']:
                continue
            all_tags.add(gold_tag)
            for pred_tag in confusion_matrix[gold_tag]:
                all_tags.add(pred_tag)
        all_tags = sorted(all_tags)
        
        error_df = analyze_common_errors(confusion_matrix, all_tags, model_name, error_path)
        
        print(f"Análise completa para {model_name}. Resultados salvos em {RESULTS_DIR}")
    
    # Gera relatório comparativo final usando as acurácias do relatório principal
    if model_accuracies:
        generate_comparative_report_from_accuracies(model_accuracies)
    else:
        # Caso não tenha sido possível extrair as acurácias, usa o método original
        generate_comparative_report()

def generate_comparative_report():
    """Gera um relatório comparativo entre todos os modelos."""
    error_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("error_analysis_") and f.endswith(".txt")]
    
    if not error_files:
        print("Nenhum arquivo de análise de erros encontrado.")
        return
    
    # Extrai informações relevantes de cada arquivo
    model_summary = []
    
    for error_file in error_files:
        model_name = error_file.replace("error_analysis_", "").replace(".txt", "")
        file_path = os.path.join(RESULTS_DIR, error_file)
        
        # Extrai as taxas de erro por tag
        error_rates = {}
        accuracy = None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Procura pela acurácia
            accuracy_pattern = r"Acurácia.*?(\d+\.\d+)"
            accuracy_match = re.search(accuracy_pattern, content)
            accuracy = float(accuracy_match.group(1)) if accuracy_match else 0.0
            
            # Extrai as taxas de erro por tag
            error_rate_pattern = r"Erros por Categoria de Tag:.*?\n(.*?)(?:\n\n|\Z)"
            match = re.search(error_rate_pattern, content, re.DOTALL)
            
            if match:
                error_block = match.group(1)
                lines = error_block.strip().split('\n')
                
                # Pula o cabeçalho e a linha de separação
                for line in lines[2:]:  # Pular cabeçalho e a linha de traços
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 4:
                            tag = parts[0]
                            total = int(parts[1])
                            errors = int(parts[2])
                            error_rate = float(parts[3])
                            
                            if total > 50:  # Apenas tags com número significativo de ocorrências
                                error_rates[tag] = error_rate
        
        # Adiciona ao resumo
        model_parts = model_name.split('_')
        model_type = model_parts[0] if len(model_parts) > 0 else ""
        smoothing = model_parts[1] if len(model_parts) > 1 else ""
        
        model_summary.append({
            'Model': model_name,
            'Type': model_type,
            'Smoothing': smoothing,
            'Accuracy': accuracy,
            'Error Rates': error_rates
        })
    
    # Se não temos dados suficientes ou completos, obtém informações do arquivo de relatório principal
    if not model_summary or any(m['Accuracy'] is None for m in model_summary):
        try:
            report_file = os.path.join(RESULTS_DIR, "report.txt")
            if os.path.exists(report_file):
                with open(report_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Extrai informações de acurácia para cada modelo/smoothing
                    models = {}
                    
                    # Padrão para capturar as tabelas de acurácia
                    test_pattern = r"Conjunto de Teste:.*?Modelo\s+Smoothing\s+Acurácia\s+\n[-]+\n(.*?)(?:\n\n|\Z)"
                    test_match = re.search(test_pattern, content, re.DOTALL)
                    
                    if test_match:
                        test_block = test_match.group(1)
                        for line in test_block.strip().split('\n'):
                            parts = line.split()
                            if len(parts) >= 3:
                                model_type = parts[0]
                                smoothing = parts[1]
                                accuracy = float(parts[2])
                                
                                model_name = f"{model_type}_{smoothing}"
                                models[model_name] = {'Accuracy': accuracy, 'Type': model_type, 'Smoothing': smoothing}
                    
                    # Atualiza o model_summary com os dados do relatório
                    for i, model_data in enumerate(model_summary):
                        model_name = model_data['Model']
                        if model_name in models and model_data['Accuracy'] is None:
                            model_summary[i]['Accuracy'] = models[model_name]['Accuracy']
                    
                    # Adiciona modelos ausentes
                    for model_name, model_data in models.items():
                        if not any(m['Model'] == model_name for m in model_summary):
                            model_summary.append({
                                'Model': model_name,
                                'Type': model_data['Type'],
                                'Smoothing': model_data['Smoothing'],
                                'Accuracy': model_data['Accuracy'],
                                'Error Rates': {}
                            })
        except Exception as e:
            print(f"Erro ao processar o arquivo de relatório: {e}")
    
    # Gera o relatório comparativo
    with open(os.path.join(RESULTS_DIR, "comparative_analysis.txt"), 'w', encoding='utf-8') as f:
        f.write("ANÁLISE COMPARATIVA DE MODELOS POS TAGGER\n")
        f.write("=" * 60 + "\n\n")
        
        # Compara acurácias
        f.write("COMPARAÇÃO DE ACURÁCIAS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Modelo':<20} {'Tipo':<10} {'Smoothing':<15} {'Acurácia':<10}\n")
        f.write("-" * 60 + "\n")
        
        # Ordena por acurácia
        model_summary.sort(key=lambda x: x['Accuracy'] if x['Accuracy'] is not None else 0, reverse=True)
        
        for model in model_summary:
            accuracy = f"{model['Accuracy']:.4f}" if model['Accuracy'] is not None else "N/A"
            f.write(f"{model['Model']:<20} {model['Type']:<10} {model['Smoothing']:<15} {accuracy}\n")
        
        # Compara taxas de erro para tags comuns
        f.write("\n\nCOMPARAÇÃO DE TAXAS DE ERRO POR TAG:\n")
        f.write("-" * 60 + "\n")
        
        # Encontra todas as tags comuns
        common_tags = set()
        for model in model_summary:
            for tag in model.get('Error Rates', {}):
                common_tags.add(tag)
        
        if common_tags:
            # Escreve o cabeçalho
            f.write(f"{'Tag':<10}")
            for model in model_summary:
                f.write(f" {model['Model']:<15}")
            f.write("\n")
            f.write("-" * (10 + 15 * len(model_summary)) + "\n")
            
            # Escreve as taxas de erro por tag
            for tag in sorted(common_tags):
                f.write(f"{tag:<10}")
                for model in model_summary:
                    error_rate = model.get('Error Rates', {}).get(tag, None)
                    value = f"{error_rate:.4f}" if error_rate is not None else "N/A"
                    f.write(f" {value:<15}")
                f.write("\n")
        
        # Análise de tendências
        f.write("\n\nANÁLISE DE TENDÊNCIAS:\n")
        f.write("-" * 60 + "\n")
        
        # Agrupa modelos por tipo
        by_type = defaultdict(list)
        for model in model_summary:
            if model['Type']:  # Verifica se o tipo não está vazio
                by_type[model['Type']].append(model)
        
        # Analisa impacto do tipo de modelo
        f.write("1. Impacto do Tipo de Modelo:\n")
        type_accs = []
        for model_type, models in by_type.items():
            valid_models = [m for m in models if m['Accuracy'] is not None]
            if valid_models:
                avg_acc = sum(m['Accuracy'] for m in valid_models) / len(valid_models)
                type_accs.append((model_type, avg_acc))
                f.write(f"   - {model_type}: Acurácia média de {avg_acc:.4f}\n")
        
        # Agrupa modelos por método de smoothing
        by_smoothing = defaultdict(list)
        for model in model_summary:
            if model['Smoothing']:  # Verifica se o smoothing não está vazio
                by_smoothing[model['Smoothing']].append(model)
        
        # Analisa impacto do método de smoothing
        f.write("\n2. Impacto do Método de Smoothing:\n")
        smooth_accs = []
        for smoothing, models in by_smoothing.items():
            valid_models = [m for m in models if m['Accuracy'] is not None]
            if valid_models:
                avg_acc = sum(m['Accuracy'] for m in valid_models) / len(valid_models)
                smooth_accs.append((smoothing, avg_acc))
                f.write(f"   - {smoothing}: Acurácia média de {avg_acc:.4f}\n")
        
        # Conclusões
        f.write("\n\nCONCLUSÕES:\n")
        f.write("-" * 60 + "\n")
        
        # Melhor modelo
        best_model = model_summary[0] if model_summary else None
        if best_model and best_model['Accuracy'] is not None:
            f.write(f"1. Melhor modelo: {best_model['Model']} com acurácia de {best_model['Accuracy']:.4f}\n")
        
        # Impacto do tipo de modelo
        type_accs.sort(key=lambda x: x[1], reverse=True)
        
        if type_accs:
            f.write(f"\n2. Modelos do tipo {type_accs[0][0]} tiveram o melhor desempenho médio ({type_accs[0][1]:.4f})\n")
        
        # Impacto do método de smoothing
        smooth_accs.sort(key=lambda x: x[1], reverse=True)
        
        if smooth_accs:
            f.write(f"\n3. O método de smoothing {smooth_accs[0][0]} teve o melhor desempenho médio ({smooth_accs[0][1]:.4f})\n")
        
        # Tags problemáticas
        if common_tags:
            problematic_tags = []
            for tag in common_tags:
                # Calcula a média das taxas de erro para essa tag
                rates = []
                for model in model_summary:
                    rate = model.get('Error Rates', {}).get(tag, None)
                    if rate is not None:
                        rates.append(rate)
                
                if rates:
                    avg_rate = sum(rates) / len(rates)
                    
                    if avg_rate > 0.2:  # Tags com taxa de erro média acima de 20%
                        problematic_tags.append((tag, avg_rate))
            
            problematic_tags.sort(key=lambda x: x[1], reverse=True)
            
            if problematic_tags:
                f.write("\n4. Tags mais problemáticas (maior taxa de erro média):\n")
                for tag, rate in problematic_tags[:5]:
                    f.write(f"   - {tag}: {rate:.4f}\n")
        
        # Recomendações finais
        f.write("\n5. Recomendações para melhoria:\n")
        if type_accs:
            f.write(f"   - Use modelos de tipo {type_accs[0][0]}\n")
        if smooth_accs:
            f.write(f"   - Prefira smoothing do tipo {smooth_accs[0][0]}\n")
        f.write("   - Para melhorar o desempenho geral, foque nas tags problemáticas identificadas\n")
        f.write("   - Considere features adicionais para ajudar na distinção de tags confundidas\n")
        f.write("   - Explore técnicas de smoothing mais avançadas\n")
    
    print(f"Análise comparativa completa. Resultados salvos em {os.path.join(RESULTS_DIR, 'comparative_analysis.txt')}")

def generate_comparative_report_from_accuracies(model_accuracies):
    """Gera um relatório comparativo a partir das acurácias extraídas."""
    
    # Organiza os modelos para o relatório
    model_summary = []
    for model_name, accuracy in model_accuracies.items():
        # Extrai o tipo de modelo e smoothing a partir do nome
        model_parts = model_name.split('_')
        model_type = model_parts[0] if len(model_parts) > 0 else ""
        smoothing = model_parts[1] if len(model_parts) > 1 else ""
        
        model_summary.append({
            'Model': model_name,
            'Type': model_type,
            'Smoothing': smoothing,
            'Accuracy': accuracy
        })
    
    # Gera o relatório comparativo
    with open(os.path.join(RESULTS_DIR, "comparative_analysis.txt"), 'w', encoding='utf-8') as f:
        f.write("ANÁLISE COMPARATIVA DE MODELOS POS TAGGER\n")
        f.write("=" * 60 + "\n\n")
        
        # Compara acurácias
        f.write("COMPARAÇÃO DE ACURÁCIAS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Modelo':<20} {'Tipo':<10} {'Smoothing':<15} {'Acurácia':<10}\n")
        f.write("-" * 60 + "\n")
        
        # Ordena por acurácia
        model_summary.sort(key=lambda x: x['Accuracy'] if x['Accuracy'] is not None else 0, reverse=True)
        
        for model in model_summary:
            accuracy = f"{model['Accuracy']:.4f}" if model['Accuracy'] is not None else "N/A"
            f.write(f"{model['Model']:<20} {model['Type']:<10} {model['Smoothing']:<15} {accuracy}\n")
        
        # Agrupa modelos por tipo
        by_type = defaultdict(list)
        for model in model_summary:
            if model['Type']:  # Verifica se o tipo não está vazio
                by_type[model['Type']].append(model)
        
        # Análise de tendências
        f.write("\n\nANÁLISE DE TENDÊNCIAS:\n")
        f.write("-" * 60 + "\n")
        
        # Analisa impacto do tipo de modelo
        f.write("1. Impacto do Tipo de Modelo:\n")
        type_accs = []
        for model_type, models in by_type.items():
            valid_models = [m for m in models if m['Accuracy'] is not None]
            if valid_models:
                avg_acc = sum(m['Accuracy'] for m in valid_models) / len(valid_models)
                type_accs.append((model_type, avg_acc))
                f.write(f"   - {model_type}: Acurácia média de {avg_acc:.4f}\n")
        
        # Agrupa modelos por método de smoothing
        by_smoothing = defaultdict(list)
        for model in model_summary:
            if model['Smoothing']:  # Verifica se o smoothing não está vazio
                by_smoothing[model['Smoothing']].append(model)
        
        # Analisa impacto do método de smoothing
        f.write("\n2. Impacto do Método de Smoothing:\n")
        smooth_accs = []
        for smoothing, models in by_smoothing.items():
            valid_models = [m for m in models if m['Accuracy'] is not None]
            if valid_models:
                avg_acc = sum(m['Accuracy'] for m in valid_models) / len(valid_models)
                smooth_accs.append((smoothing, avg_acc))
                f.write(f"   - {smoothing}: Acurácia média de {avg_acc:.4f}\n")
        
        # Conclusões
        f.write("\n\nCONCLUSÕES:\n")
        f.write("-" * 60 + "\n")
        
        # Melhor modelo
        best_model = model_summary[0] if model_summary else None
        if best_model and best_model['Accuracy'] is not None:
            f.write(f"1. Melhor modelo: {best_model['Model']} com acurácia de {best_model['Accuracy']:.4f}\n")
        
        # Impacto do tipo de modelo
        type_accs.sort(key=lambda x: x[1], reverse=True)
        
        if type_accs:
            f.write(f"\n2. Modelos do tipo {type_accs[0][0]} tiveram o melhor desempenho médio ({type_accs[0][1]:.4f})\n")
        
        # Impacto do método de smoothing
        smooth_accs.sort(key=lambda x: x[1], reverse=True)
        
        if smooth_accs:
            f.write(f"\n3. O método de smoothing {smooth_accs[0][0]} teve o melhor desempenho médio ({smooth_accs[0][1]:.4f})\n")
        
        # Recomendações finais
        f.write("\n4. Recomendações para melhoria:\n")
        if type_accs:
            f.write(f"   - Use modelos de tipo {type_accs[0][0]}\n")
        if smooth_accs:
            f.write(f"   - Prefira smoothing do tipo {smooth_accs[0][0]}\n")
        f.write("   - Para melhorar o desempenho geral, explore features adicionais\n")
        f.write("   - Considere técnicas de smoothing mais avançadas\n")
    
    print(f"Análise comparativa baseada em acurácias completa. Resultados salvos em {os.path.join(RESULTS_DIR, 'comparative_analysis.txt')}")
    

if __name__ == "__main__":
    # Verificar se há argumentos para processar apenas modelos específicos
    if len(sys.argv) > 1 and sys.argv[1] == "--only-matrices":
        # Gera apenas as matrizes para os modelos especificados
        result_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("detailed_") and f.endswith(".txt")]
        
        if not result_files:
            print("Nenhum arquivo de resultados detalhados encontrado.")
            sys.exit(1)
        
        for result_file in result_files:
            # Extrai nome do modelo da nomenclatura do arquivo
            model_name = result_file.replace("detailed_", "").replace(".txt", "")
            
            print(f"Gerando matrizes para modelo: {model_name}")
            
            # Caminho completo para o arquivo
            file_path = os.path.join(RESULTS_DIR, result_file)
            
            # Extrai a matriz de confusão
            confusion_matrix = extract_confusion_matrix(file_path)
            
            if not confusion_matrix:
                print(f"Não foi possível extrair a matriz de confusão para {model_name}")
                continue
            
            # Gera apenas a matriz simplificada
            simplified_path = os.path.join(RESULTS_DIR, f"matrix_simplificada_{model_name}.png")
            simplified_matrix, simplified_tags = plot_simplified_confusion_matrix(confusion_matrix, model_name, simplified_path)
            
            print(f"Matriz simplificada gerada para {model_name}")
    else:
        # Executa a análise completa
        analyze_all_models()
