import pandas as pd
import numpy as np
from pathlib import Path

# Modelos, categorias, métodos e métricas
models = ['deepseek-14b', 'deepseek-8b', 'gemma2_9b', 'gemma_7b', 'gpt-4o-mini',
          'llama3.1_8b', 'llama3.2_3b', 'mistral-nemo_12b', 'mistral_7b', 'phi4_14b']

original_categories = [
    "Bitter Frustration", "Impatience", "Vulgarity",
    "Irony", "Identify Attack/Name Calling", "Threat",
    "Insulting", "Entitlement", "Mocking", "None"
]
methods = ["Zero-shot", "One-shot", "Few-shot", "Auto-CoT", "Role-based"]
metrics = ["Pr", "Re", "F1"]

# Renomear apenas no MultiIndex final
incivility_rename_map = {
    "Identify Attack/Name Calling": "Identify Attack"
}
export_categories = [incivility_rename_map.get(cat, cat).lower() for cat in original_categories]

# Mapas de nomes
strategy_map = {
    "zero_shot": "Zero-shot",
    "one_shot": "One-shot",
    "few_shot_3": "Few-shot",
    "auto_cot": "Auto-CoT",
    "role_based": "Role-based"
}

model_name_map = {
    'gemma_7b': 'gemma:7b',
    'gemma2_9b': 'gemma2:9b',
    'mistral-nemo_12b': 'mistral-nemo:12b',
    'mistral_7b': 'mistral:7b',
    'deepseek-8b': 'deepseek-r1:8b',
    'deepseek-14b': 'deepseek-r1:14b',
    'llama3.2_3b': 'llama3.2:3b',
    'llama3.1_8b': 'llama3.1:8b',
    'phi4_14b': 'phi4:14b',
    'gpt-4o-mini': 'gpt-4o-mini',
    'Average': 'Average'
}

# Leitura do CSV
results_table_path = Path('results_table')
results_table_path.mkdir(parents=True, exist_ok=True)
df_real = pd.read_csv(results_table_path / "results_concat.csv")

# Renomear colunas e estratégia
df_real['Strategy'] = df_real['Strategy'].map(strategy_map)
df_real = df_real.rename(columns={
    'Modelo': 'Model',
    'Tbdf': 'Category',
    'Precision': 'Pr',
    'Recall': 'Re',
    'F1-score': 'F1'
})
df_real['Category'] = df_real['Category'].str.lower()

# Criar estrutura do dataframe final com nomes "exportáveis"
columns = pd.MultiIndex.from_product([export_categories, methods, metrics])
df_final = pd.DataFrame(index=models, columns=columns)

# Preencher os dados
for _, row in df_real.iterrows():
    model = row['Model']
    cat = row['Category']
    strat = row['Strategy']
    if model in models and cat in [c.lower() for c in original_categories] and strat in methods:
        export_cat = incivility_rename_map.get(cat.title(), cat.title()).lower()
        for metric in metrics:
            df_final.loc[model, (export_cat, strat, metric)] = row[metric]

# Conversão e média
df_final = df_final.astype(float)
df_final = df_final.round(2)
df_final.loc['Average'] = df_final.mean(numeric_only=True)

# Renomear modelos na exportação
df_final.rename(index=model_name_map, inplace=True)

# Exportar para Excel
path = results_table_path / 'rq2.xlsx'
df_final.to_excel(path, index=True, sheet_name='RQ2', engine='openpyxl')
