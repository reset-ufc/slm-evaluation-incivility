from pathlib import Path
import pandas as pd
from collections import Counter

results_table_path = Path('results_table')

results_concat = results_table_path / 'results_concat.csv'

df = pd.read_csv(results_concat)

df = df.loc[~df['Strategy'].str.contains('role_based_')]

df['Precision'] = df['Precision'].round(2)
df['Recall'] = df['Recall'].round(2)

# print(df)

classes = df['Tbdf'].unique()
best_configs = []
def get_best_model():
    for cls in classes:
        df_cls = df.loc[df['Tbdf'] == cls].reset_index()
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


    print("Tamanho do dicionario de melhores configs:", len(best_configs))

    contagem = Counter(best_configs).most_common(3)

    print('Best config:', contagem[0][0], 'aparecendo', contagem[0][1], 'vezes')

    print('Second Best config:', contagem[1][0], 'aparecendo', contagem[1][1], 'vezes')

    print('Third Best config:', contagem[2][0], 'aparecendo', contagem[2][1], 'vezes')

    return contagem[0][0]
get_best_model()