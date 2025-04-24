import pandas as pd
import numpy as np
import json
from pathlib import Path

data_path = Path('data')
df_path = data_path / "final_df_cleaned.csv"
 
# Load the data
df = pd.read_csv(df_path)

# Select examples civil
civil_df = df[df['actual'] == 0].reset_index(drop=True)
civil_examples = civil_df.loc[np.random.choice(len(civil_df), 3, replace=False), 'message'].tolist()

# Select examples uncivil
uncivil_df = df[df['actual'] == 1].reset_index(drop=True)
uncivil_examples = uncivil_df.loc[np.random.choice(len(uncivil_df), 3, replace=False), 'message'].tolist()

# Build the dictionary with examples
data = {
    "civil": civil_examples,
    "uncivil": uncivil_examples
}

# JSON file path
file_path = data_path / 'exemplos.json'

# Save the dictionary to a JSON file
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

