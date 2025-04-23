import pandas as pd 
import numpy as np
from pathlib import Path

def get_rq1_table():
    results_path = Path('results')
    compact_table_path_uncivil = results_path / 'compact_result_table_uncivil_without_duplicates.xlsx'
    compact_table_path_civil = results_path / 'compact_result_table_civil_without_duplicates.xlsx'

    uncivil = pd.read_excel(compact_table_path_uncivil)
    uncivil = uncivil.ffill()
    uncivil = uncivil.loc[~uncivil['Model'].isin(['toxicr', 'refined_model']).values]
    uncivil = uncivil.rename(columns={'precision': 'Precision', 'recall': 'Recall', 'f1-score': 'F1-score'})
    uncivil_sem_combinacoes = uncivil.loc[~uncivil['Strategy'].str.contains('role_based_')].copy()  # excluir combinações com role-based

    civil = pd.read_excel(compact_table_path_civil)
    civil = civil.ffill()
    civil = civil.loc[~civil['Model'].isin(['toxicr', 'refined_model']).values]
    civil = civil.rename(columns={'precision': 'Precision', 'recall': 'Recall', 'f1-score': 'F1-score'})
    civil_sem_combinacoes = civil.loc[~civil['Strategy'].str.contains('role_based_')].copy()  # excluir combinações com role-based

    uncivil_data = uncivil_sem_combinacoes.groupby('Model')[['F1-score', 'Precision', 'Recall']].mean().reset_index().copy()
    civil_data = civil_sem_combinacoes.groupby('Model')[['F1-score', 'Precision', 'Recall']].mean().reset_index().copy()

    models = ['phi4:14b', 'deepseek-r1:14b', 'mistral-nemo:12b', 'gemma2:9b', 'llama3.1:8b', 'deepseek-r1:8b', 'gemma:7b', 'mistral:7b', 'llama3.2:3b', 'gpt-4o-mini']

    models = [m.replace(':', '_') for m in models]

    uncivil_data = uncivil_data.set_index('Model').loc[models, ['Precision', 'Recall', 'F1-score']].reset_index()
    civil_data = civil_data.set_index('Model').loc[models, ['Precision', 'Recall', 'F1-score']].reset_index()
    
    uncivil_data = uncivil_sem_combinacoes.groupby('Model')[['Precision', 'Recall', 'F1-score']].mean().reset_index().copy()
    civil_data = civil_sem_combinacoes.groupby('Model')[['Precision', 'Recall', 'F1-score']].mean().reset_index().copy()
    
    categories = ['Civil', 'Uncivil']
    with pd.ExcelWriter(results_path / 'rq1_table.xlsx') as writer:
        for i, table in enumerate([civil_data, uncivil_data]):
            table.to_excel(writer, sheet_name=categories[i])

get_rq1_table()
