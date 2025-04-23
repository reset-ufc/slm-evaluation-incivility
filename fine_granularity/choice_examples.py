import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load the dataset
data_path = Path('data')
df = pd.read_csv(data_path / 'reference_dataset_fg.csv')

# Function to select examples from each class
def selecionar_exemplos(df, classe, n=3):
    subset = df[df['tbdf'] == classe]
    if len(subset) >= n:
        return subset.sample(n)['comment_body'].tolist()
    else:
        print(f"[Aviso] Classe '{classe}' tem apenas {len(subset)} exemplos.")
        return subset['comment_body'].tolist()

# Criar o dicionário final
data = {
    "bitter frustration": selecionar_exemplos(df, 'bitter frustration'),
    "impatience": selecionar_exemplos(df, 'impatience'),
    "mocking": selecionar_exemplos(df, 'mocking'),
    "entitlement": selecionar_exemplos(df, 'entitlement'),
    "irony": selecionar_exemplos(df, 'irony'),
    "vulgarity": selecionar_exemplos(df, 'vulgarity'),
    "insulting": selecionar_exemplos(df, 'insulting'),
    "threat": selecionar_exemplos(df, 'threat'),
    "identify attack/name calling": selecionar_exemplos(df, 'identify attack/name calling'),  # nome certo do CSV
    "none": selecionar_exemplos(df, 'none')
}

# Caminho do arquivo JSON
file_path = data_path / 'examples_fg.json'

# Salvar como JSON
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
