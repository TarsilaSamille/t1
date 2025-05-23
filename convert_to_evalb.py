#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para converter dados do formato de tag (palavra_TAG) para o formato
do evalb utilizado na avaliação de PoS tagging:
(S (tag1 word1) (tag2 word2) (tag3 word3) ... (tagn wordn) )
"""

import sys
import os
import re

def convert_to_evalb_format(input_file, output_file):
    """
    Converte um arquivo do formato palavra_TAG para o formato evalb.
    
    Args:
        input_file (str): Caminho para o arquivo de entrada no formato palavra_TAG
        output_file (str): Caminho para o arquivo de saída no formato evalb
    """
    print(f"Convertendo {input_file} para formato evalb...")
    
    # Define tags de pontuação que não devem ser contabilizadas
    PUNCTUATION_TAGS = set(['.',',','``',"''",':','(',')','-LRB-','-RRB-',';','--','-',])
    
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            
            # Processa a linha para extrair pares palavra-tag
            tokens = []
            for token in line.split():
                if '_' in token:
                    word, tag = token.rsplit('_', 1)
                    
                    # Ignora BOS e EOS na avaliação
                    if tag != "<s>" and tag != "</s>":
                        tokens.append((word, tag))
            
            # Converte para o formato evalb (ignorando pontuação na contagem)
            evalb_line = "(S"
            for word, tag in tokens:
                evalb_line += f" ({tag} {word})"
            evalb_line += " )"
            
            outfile.write(evalb_line + "\n")
    
    print(f"Conversão concluída. Arquivo salvo em {output_file}")

def main():
    if len(sys.argv) < 3:
        print("Uso: python convert_to_evalb.py <arquivo_entrada> <arquivo_saida>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Erro: Arquivo {input_file} não encontrado.")
        sys.exit(1)
    
    convert_to_evalb_format(input_file, output_file)

if __name__ == "__main__":
    main()
