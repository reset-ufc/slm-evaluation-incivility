import pandas as pd
import numpy as np
from pathlib import Path

models = ['deepseek-14b', 'deepseek-8b', 'gemma2_9b', 'gemma_7b', 'gpt-4o-mini',
 'llama3.1_8b', 'llama3.2_3b', 'mistral-nemo_12b', 'mistral_7b', 'phi4_14b']
categories = [
    "Bitter Frustration", "Impatience", "Vulgarity",
    "Irony", "Identify Attack/Name Calling", "Threat",
    "Insulting", "Entitlement", "Mocking"
]
categories = [cat.lower() for cat in categories]  # Normalize to lowercase
methods = ["Zero-shot", "One-shot", "Few-shot", "Auto-CoT", "Role-based"]
metrics = ["Pr", "Re", "F1"]


# Mapping for strategy names
# This is a dictionary that maps the strategy names in the CSV to more readable names.
strategy_map = {
    "zero_shot": "Zero-shot",
    "one_shot": "One-shot",
    "few_shot_3": "Few-shot",
    "auto_cot": "Auto-CoT",
    "role_based": "Role-based"
}

# Load the CSV file with results
results_table_path = Path('results_table')
results_table_path.mkdir(parents=True, exist_ok=True)
df_real = pd.read_csv(results_table_path / "results_concat.csv")

# Rename columns for consistency
df_real['Strategy'] = df_real['Strategy'].map(strategy_map)
df_real = df_real.rename(columns={
    'Modelo': 'Model',
    'Tbdf': 'Category',
    'Precision': 'Pr',
    'Recall': 'Re',
    'F1-score': 'F1'
})

# Build the DataFrame structure with MultiIndex
columns = pd.MultiIndex.from_product([categories, methods, metrics])
df_final = pd.DataFrame(index=models, columns=columns)

# Fill the DataFrame with values from df_real
for _, row in df_real.iterrows():
    model = row['Model']
    cat = row['Category']
    strat = row['Strategy']
    
    if model in models and cat in categories and strat in methods:
        for metric in metrics:
            df_final.loc[model, (cat, strat, metric)] = row[metric]

# Convert to float
df_final = df_final.astype(float)

# Add the average row
df_final.loc['Average'] = df_final.mean(numeric_only=True)

# Export to Excel
path = results_table_path / 'rq2.xlsx'
df_final.to_excel(path, index=True, sheet_name='RQ2', engine='openpyxl')
