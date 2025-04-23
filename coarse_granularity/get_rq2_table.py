import pandas as pd 
import numpy as np
from pathlib import Path

def get_rq2_table(results_path):
    compact_table_uncivil_path = results_path / 'compact_result_table_uncivil_without_duplicates.xlsx'
    compact_table_civil_path = results_path / 'compact_result_table_civil_without_duplicates.xlsx'

    civil_compact = pd.read_excel(compact_table_civil_path)
    civil_compact['Strategy'] = civil_compact['Strategy'].ffill()
    uncivil_compact = pd.read_excel(compact_table_uncivil_path)
    uncivil_compact['Strategy'] = uncivil_compact['Strategy'].ffill()

    models = ['phi4:14b', 'deepseek-r1:14b', 'mistral-nemo:12b', 'gemma2:9b', 'llama3.1:8b', 'deepseek-r1:8b', 'gemma:7b', 'mistral:7b', 'llama3.2:3b', 'gpt-4o-mini']

    models = [m.replace(':', '_') for m in models]
    strategies = ['zero_shot', 'one_shot', 'few_shot', 'auto_cot', 'role_based']
    metrics = ['precision', 'recall', 'f1-score']

    columns = pd.MultiIndex.from_product([strategies, metrics], names=['Scenario', 'Metric'])

    df_civil = pd.DataFrame(index=models, columns=columns)
    df_uncivil = pd.DataFrame(index=models, columns=columns)

    for i, row in civil_compact.iterrows():
        model = row['Model']
        strategy = row['Strategy']
        if model in models and strategy in strategies:
            for metric in metrics:
                df_civil.loc[model, (strategy, metric)] = row[metric]
    
    for i, row in uncivil_compact.iterrows():
        model = row['Model']
        strategy = row['Strategy']
        if model in models and strategy in strategies:
            for metric in metrics:
                df_uncivil.loc[model, (strategy, metric)] = row[metric]

    categories = ['Civil', 'Uncivil']
    with pd.ExcelWriter(results_path / 'rq2_table.xlsx') as writer:
        for i, table in enumerate([df_civil, df_uncivil]):
            table.to_excel(writer, sheet_name=categories[i])

results_path = Path('results')  

get_rq2_table(results_path)