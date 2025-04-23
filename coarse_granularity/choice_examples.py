import pandas as pd
import numpy as np
import json
from pathlib import Path

data_path = Path('data')
df_path = data_path / "final_df_cleaned.csv"
 
# Carregar o DataFrame
df = pd.read_csv(df_path)

# Selecionar exemplos civil
civil_df = df[df['actual'] == 0].reset_index(drop=True)
civil_examples = civil_df.loc[np.random.choice(len(civil_df), 3, replace=False), 'message'].tolist()

# Selecionar exemplos uncivil
uncivil_df = df[df['actual'] == 1].reset_index(drop=True)
uncivil_examples = uncivil_df.loc[np.random.choice(len(uncivil_df), 3, replace=False), 'message'].tolist()

# Criar um dicionário estruturado
data = {
    "civil": civil_examples,
    "uncivil": uncivil_examples
}

# Caminho do arquivo JSON
file_path = r'C:\Users\mario\Documents\estudos\ufc\Granularidade Grossa\data\exemplos.json'

# Salvar como JSON
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

