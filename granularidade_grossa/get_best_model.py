from pathlib import Path
import pandas as pd
from collections import Counter

results_path = Path('results')
compact_table_path_uncivil = results_path / 'compact_result_table_uncivil_without_duplicates.xlsx'
compact_table_path_civil = results_path / 'compact_result_table_civil_without_duplicates.xlsx'

uncivil = pd.read_excel(compact_table_path_uncivil)
uncivil = uncivil.ffill()
uncivil = uncivil.loc[~uncivil['Model'].isin(['toxicr', 'refined_model']).values]
uncivil = uncivil.rename(columns={'precision': 'Precision', 'recall': 'Recall', 'f1-score': 'F1-score', 'Model': 'Modelo'})
uncivil_sem_combinacoes = uncivil.loc[~uncivil['Strategy'].str.contains('role_based_')].copy()  # excluir combinações com role-based

civil = pd.read_excel(compact_table_path_civil)
civil = civil.ffill()
civil = civil.loc[~civil['Model'].isin(['toxicr', 'refined_model']).values]
civil = civil.rename(columns={'precision': 'Precision', 'recall': 'Recall', 'f1-score': 'F1-score', 'Model': 'Modelo'})
civil_sem_combinacoes = civil.loc[~civil['Strategy'].str.contains('role_based_')].copy()

uncivil_sem_combinacoes["Class"] = "Uncivil"
civil_sem_combinacoes["Class"] = "Civil"

df = pd.concat([uncivil_sem_combinacoes, civil_sem_combinacoes])

df['Precision'] = df['Precision'].round(2)
df['Recall'] = df['Recall'].round(2)


classes = df['Class'].unique()
best_configs = []

for cls in classes:
    df_cls = df.loc[df['Class'] == cls].reset_index()
    best_config_cls_f1 = df_cls.loc[df_cls['F1-score'] == df_cls['F1-score'].max(), :]
    best_config_cls_pr = df_cls.loc[df_cls['Precision'] == df_cls['Precision'].max(), :]

    for i, config in best_config_cls_f1.iterrows():
        model = config['Modelo']
        strategy = config['Strategy']

        best_configs.append(f'{model} + {strategy}')
    
    for i, config in best_config_cls_pr.iterrows():
        model = config['Modelo']
        strategy = config['Strategy']

        best_configs.append(f'{model} + {strategy}')

    # best_model_cls_f1 = best_config_cls_f1.loc[:, 'Modelo']
    # best_strategy_cls_f1 = best_config_cls_f1.loc[:, 'Strategy']

    # best_model_cls_pr = best_config_cls_pr.loc[:, 'Modelo']
    # best_strategy_cls_pr = best_config_cls_pr.loc[:, 'Strategy']

    # print(f'On {cls}, the best model was {best_model_cls_f1} and the best strategy was {best_strategy_cls_f1} on F1-score')
    # print(f'On {cls}, the best model was {best_model_cls_pr} and the best strategy was {best_strategy_cls_pr} on Precision')

    # best_configs[cls + ' F1'] = f'{best_model_cls_f1} + {best_strategy_cls_f1}'
    # best_configs[cls + ' Pr'] = f'{best_model_cls_pr} + {best_strategy_cls_pr}'

print("Tamanho do dicionario de melhores configs:", len(best_configs))

contagem = Counter(best_configs).most_common(3)

print('Best config:', contagem[0][0], 'aparecendo', contagem[0][1], 'vezes')

print('Second Best config:', contagem[1][0], 'aparecendo', contagem[1][1], 'vezes')

print('Third Best config:', contagem[2][0], 'aparecendo', contagem[2][1], 'vezes')

