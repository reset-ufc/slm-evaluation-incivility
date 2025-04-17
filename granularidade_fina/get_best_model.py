from pathlib import Path
import pandas as pd
from collections import Counter

results_table_path = Path('results_table')

results_concat = results_table_path / 'results_concat.csv'

df = pd.read_csv(results_concat)

df = df.loc[~df['Strategy'].str.contains('role_based_')]

df['Precision'] = df['Precision'].round(2)
df['Recall'] = df['Recall'].round(2)

print(df)

classes = df['Tbdf'].unique()
best_configs = []

for cls in classes:
    df_cls = df.loc[df['Tbdf'] == cls].reset_index()
    best_config_cls_f1 = df_cls.loc[df_cls['F1-score'] == df_cls['F1-score'].max(), :]
    best_config_cls_pr = df_cls.loc[df_cls['Precision'] == df_cls['Precision'].max(), :]
    # print(len(best_config_cls_f1))
    # print(len(best_config_cls_pr))

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

