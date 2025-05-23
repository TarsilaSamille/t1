#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para executar experimentos com o POS Tagger em diferentes configurações
e gerar relatórios comparativos dos resultados.
"""

import os
import sys
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pandas as pd
import seaborn as sns
from datetime import datetime

# Configurações dos experimentos
MODELS = ["unigram", "bigram", "trigram"]
SMOOTHING_METHODS = ["backoff", "interpolation", "none"]

# Caminhos dos arquivos
TRAIN_FILE = "Secs0-18"
DEV_FILE = "Secs19-21"
TEST_FILE = "Secs22-24"

# Diretório para salvar resultados
RESULTS_DIR = "results"

def run_experiments():
    """Executa todos os experimentos definidos."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Resultados para guardar acurácias
    results = {}
    
    # Para cada combinação de modelo e método de smoothing
    for model in MODELS:
        for smoothing in SMOOTHING_METHODS:
            print(f"\n{'='*60}")
            print(f"Executando experimento: Modelo {model}, Smoothing {smoothing}")
            print(f"{'='*60}")
            
            # Nome do experimento
            exp_name = f"{model}_{smoothing}"
            
            # Treina o modelo
            print("\nTreinando modelo...")
            train_cmd = f"python pos_tagger.py train {TRAIN_FILE} {model} {smoothing}"
            subprocess.run(train_cmd, shell=True)
            
            # Avalia no conjunto de desenvolvimento
            print("\nAvaliando no conjunto de desenvolvimento...")
            dev_cmd = f"python pos_tagger.py eval {DEV_FILE} {model} {smoothing}"
            dev_output = subprocess.check_output(dev_cmd, shell=True, universal_newlines=True)
            
            # Extrai a acurácia do resultado
            dev_accuracy = extract_accuracy(dev_output)
            results[(model, smoothing, "dev")] = dev_accuracy
            
            # Avalia no conjunto de teste final
            print("\nAvaliando no conjunto de teste final...")
            test_cmd = f"python pos_tagger.py eval {TEST_FILE} {model} {smoothing}"
            test_output = subprocess.check_output(test_cmd, shell=True, universal_newlines=True)
            
            # Extrai a acurácia do resultado
            test_accuracy = extract_accuracy(test_output)
            results[(model, smoothing, "test")] = test_accuracy
            
            # Salva os resultados completos
            with open(os.path.join(RESULTS_DIR, f"detailed_{exp_name}.txt"), "w") as f:
                f.write(f"Modelo: {model}, Smoothing: {smoothing}\n")
                f.write(f"{'='*60}\n\n")
                f.write("DESENVOLVIMENTO\n")
                f.write(f"{'='*60}\n")
                f.write(dev_output)
                f.write("\n\nTESTE FINAL\n")
                f.write(f"{'='*60}\n")
                f.write(test_output)
    
    # Gera relatório comparativo
    generate_report(results)
    
    return results

def extract_accuracy(output):
    """Extrai a acurácia do output do comando de avaliação."""
    for line in output.split('\n'):
        if "Acurácia final:" in line:
            return float(line.split(':')[1].strip())
    return 0.0

def generate_report(results):
    """Gera relatório comparativo dos resultados."""
    print("\n\nGerando relatório comparativo...")
    
    # Cria um DataFrame para facilitar a análise
    data = []
    for (model, smoothing, dataset), accuracy in results.items():
        data.append({
            "Modelo": model,
            "Smoothing": smoothing,
            "Dataset": dataset,
            "Acurácia": accuracy
        })
    
    df = pd.DataFrame(data)
    
    # Salva os resultados em CSV
    df.to_csv(os.path.join(RESULTS_DIR, "all_results.csv"), index=False)
    
    # Gera gráficos comparativos
    plt.figure(figsize=(12, 8))
    
    # Gráfico de barras agrupadas para desenvolvimento
    dev_df = df[df["Dataset"] == "dev"]
    ax = sns.barplot(x="Modelo", y="Acurácia", hue="Smoothing", data=dev_df)
    ax.set_title("Acurácia no Conjunto de Desenvolvimento", fontsize=14)
    ax.set_ylim(0.85, 1.0)  # Ajusta conforme necessário
    
    # Adiciona anotações com os valores
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.4f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'bottom',
                   xytext = (0, 5), textcoords = 'offset points')
    
    plt.savefig(os.path.join(RESULTS_DIR, "dev_accuracy.png"), dpi=300, bbox_inches="tight")
    
    # Gráfico de barras agrupadas para teste
    plt.figure(figsize=(12, 8))
    test_df = df[df["Dataset"] == "test"]
    ax = sns.barplot(x="Modelo", y="Acurácia", hue="Smoothing", data=test_df)
    ax.set_title("Acurácia no Conjunto de Teste", fontsize=14)
    ax.set_ylim(0.85, 1.0)  # Ajusta conforme necessário
    
    # Adiciona anotações com os valores
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.4f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'bottom',
                   xytext = (0, 5), textcoords = 'offset points')
    
    plt.savefig(os.path.join(RESULTS_DIR, "test_accuracy.png"), dpi=300, bbox_inches="tight")
    
    # Gráfico comparativo entre dev e teste
    plt.figure(figsize=(14, 10))
    
    # Reorganiza o DataFrame para facilitar o gráfico
    pivot_df = df.pivot_table(
        index=['Modelo', 'Smoothing'], 
        columns='Dataset', 
        values='Acurácia'
    ).reset_index()
    
    # Cria um DataFrame "long" para o Seaborn
    long_df = pd.melt(
        pivot_df, 
        id_vars=['Modelo', 'Smoothing'], 
        value_vars=['dev', 'test'],
        var_name='Dataset', 
        value_name='Acurácia'
    )
    
    # Plota o gráfico
    g = sns.catplot(
        x='Modelo', 
        y='Acurácia', 
        hue='Smoothing', 
        col='Dataset',
        data=long_df, 
        kind='bar', 
        height=6, 
        aspect=0.8
    )
    g.set_titles("{col_name}")
    g.set_ylabels("Acurácia")
    
    # Salva o gráfico
    plt.savefig(os.path.join(RESULTS_DIR, "compare_dev_test.png"), dpi=300, bbox_inches="tight")
    
    # Gera relatório textual
    with open(os.path.join(RESULTS_DIR, "report.txt"), "w") as f:
        f.write("RELATÓRIO DE EXPERIMENTOS POS TAGGER\n")
        f.write(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")
        
        f.write("RESUMO DE ACURÁCIA\n\n")
        
        # Tabela para desenvolvimento
        f.write("Conjunto de Desenvolvimento:\n")
        f.write(f"{'-'*60}\n")
        f.write(f"{'Modelo':<10} {'Smoothing':<15} {'Acurácia':<10}\n")
        f.write(f"{'-'*60}\n")
        
        for model in MODELS:
            for smoothing in SMOOTHING_METHODS:
                acc = results.get((model, smoothing, "dev"), 0.0)
                f.write(f"{model:<10} {smoothing:<15} {acc:.4f}\n")
        
        f.write("\n")
        
        # Tabela para teste
        f.write("Conjunto de Teste:\n")
        f.write(f"{'-'*60}\n")
        f.write(f"{'Modelo':<10} {'Smoothing':<15} {'Acurácia':<10}\n")
        f.write(f"{'-'*60}\n")
        
        for model in MODELS:
            for smoothing in SMOOTHING_METHODS:
                acc = results.get((model, smoothing, "test"), 0.0)
                f.write(f"{model:<10} {smoothing:<15} {acc:.4f}\n")
        
        # Melhor modelo
        best_model = max(
            [(model, smoothing) for model in MODELS for smoothing in SMOOTHING_METHODS],
            key=lambda x: results.get((x[0], x[1], "test"), 0.0)
        )
        
        best_acc = results.get((best_model[0], best_model[1], "test"), 0.0)
        
        f.write(f"\nMELHOR MODELO: {best_model[0]} com smoothing {best_model[1]}\n")
        f.write(f"Acurácia no teste: {best_acc:.4f}\n\n")
        
        # Análise de resultados
        f.write("ANÁLISE DE RESULTADOS\n")
        f.write(f"{'-'*60}\n\n")
        
        # Comparação de modelos
        f.write("Comparação de Modelos:\n")
        for dataset in ["dev", "test"]:
            f.write(f"\nDataset: {dataset}\n")
            for model in MODELS:
                avg_acc = sum(results.get((model, s, dataset), 0.0) for s in SMOOTHING_METHODS) / len(SMOOTHING_METHODS)
                f.write(f"Média de acurácia para modelo {model}: {avg_acc:.4f}\n")
        
        # Comparação de métodos de smoothing
        f.write("\nComparação de Métodos de Smoothing:\n")
        for dataset in ["dev", "test"]:
            f.write(f"\nDataset: {dataset}\n")
            for smoothing in SMOOTHING_METHODS:
                avg_acc = sum(results.get((m, smoothing, dataset), 0.0) for m in MODELS) / len(MODELS)
                f.write(f"Média de acurácia para smoothing {smoothing}: {avg_acc:.4f}\n")
    
    print(f"Relatório completo gerado em: {os.path.join(RESULTS_DIR, 'report.txt')}")

if __name__ == "__main__":
    run_experiments()
