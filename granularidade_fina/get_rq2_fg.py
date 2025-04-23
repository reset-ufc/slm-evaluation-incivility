import pandas as pd
import numpy as np
from pathlib import Path
# Modelos, Categorias, Estratégias e Métricas
models = ['deepseek-14b', 'deepseek-8b', 'gemma2_9b', 'gemma_7b', 'gpt-4o-mini',
 'llama3.1_8b', 'llama3.2_3b', 'mistral-nemo_12b', 'mistral_7b', 'phi4_14b']
categories = [
    "Bitter Frustration", "Impatience", "Vulgarity",
    "Irony", "Identify Attack/Name Calling", "Threat",
    "Insulting", "Entitlement", "Mocking"
]
categories = [cat.lower() for cat in categories]  # Normaliza para minúsculas
methods = ["Zero-shot", "One-shot", "Few-shot", "Auto-CoT", "Role-based"]
metrics = ["Pr", "Re", "F1"]


# Mapeamento de nomes conforme aparecem no CSV
strategy_map = {
    "zero_shot": "Zero-shot",
    "one_shot": "One-shot",
    "few_shot_3": "Few-shot",
    "auto_cot": "Auto-CoT",
    "role_based": "Role-based"
}

# Carregar CSV com dados reais
df_real = pd.read_csv("results_table/results_concat.csv")

# Padronizar nomes
df_real['Strategy'] = df_real['Strategy'].map(strategy_map)
df_real = df_real.rename(columns={
    'Modelo': 'Model',
    'Tbdf': 'Category',
    'Precision': 'Pr',
    'Recall': 'Re',
    'F1-score': 'F1'
})

# Criar a estrutura vazia
columns = pd.MultiIndex.from_product([categories, methods, metrics])
df_final = pd.DataFrame(index=models, columns=columns)

# Preencher a estrutura
for _, row in df_real.iterrows():
    model = row['Model']
    cat = row['Category']
    strat = row['Strategy']
    
    if model in models and cat in categories and strat in methods:
        for metric in metrics:
            df_final.loc[model, (cat, strat, metric)] = row[metric]

# Converter valores para float
df_final = df_final.astype(float)

# Adicionar linha de média
df_final.loc['Average'] = df_final.mean(numeric_only=True)

# Exportar para Excel
path = Path('results_table') / 'rq2.xlsx'
df_final.to_excel(path, index=True, sheet_name='RQ2', engine='openpyxl')
