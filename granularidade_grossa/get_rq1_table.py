import pandas as pd 
import numpy as np
from get_quantity_errors import get_quantity_errors

def get_rq1_table(compact_table_path):
    errors_df = get_quantity_errors()
    compact_table = pd.read_excel(compact_table_path)
    compact_table = compact_table.ffill()

    # Definindo os índices (MultiIndex)
    strategies = [
        "zero_shot", "zero_shot", 
        "few_shot", "few_shot", 
        "auto_cot", "auto_cot", 
        "role_based", "role_based"
    ]
    unique_strategies = ['zero_shot', 'few_shot', 'auto_cot', 'role_based']

    cases = [
        'Best', 'Worst',
        'Best', 'Worst',
        'Best', 'Worst',
        'Best', 'Worst'
    ]

    unique_cases = ['Best', 'Worst']

    # Criando um MultiIndex
    index = pd.MultiIndex.from_tuples(list(zip(strategies, cases)), names=["Strategy", "Case"])

    rq1_table = pd.DataFrame(index=index, columns=["precision", "recall", "f1-score", "FP", "FN", "Allucination Rate"])

    best_models_by_strategy = (compact_table.loc[compact_table.groupby('Strategy')['f1-score'].idxmax(), ['Strategy', 'Model', 'precision', 'recall', 'f1-score', 'FP', 'FN']]
                                .sort_values(by='f1-score', ascending=False))
    worst_models_by_strategy = (compact_table.loc[compact_table.groupby('Strategy')['f1-score'].idxmin(), ['Strategy', 'Model', 'precision', 'recall', 'f1-score', 'FP', 'FN']]
                                .sort_values(by='f1-score', ascending=False))
    
    for strategy in unique_strategies:
        for case in unique_cases:
            if case == 'Best':
                model = best_models_by_strategy.loc[best_models_by_strategy['Strategy'] == strategy, 'Model'].values[0]
            else:
                model = worst_models_by_strategy.loc[worst_models_by_strategy['Strategy'] == strategy, 'Model'].values[0]
            print(strategy, case, model)

            model_path = compact_table.loc[(compact_table['Strategy'] == strategy) & (compact_table['Model'] == model)]
            f1_score_value = model_path['f1-score'].values[0]
            precision_value = model_path['precision'].values[0]
            recall_value = model_path['recall'].values[0]
            fp = model_path['FP'].values[0]
            fn = model_path['FN'].values[0]

            rq1_table.loc[(strategy, case), "precision"] = f'{model} ({np.round(precision_value, 2)})'
            rq1_table.loc[(strategy, case), "recall"] = f'{model} ({np.round(recall_value, 2)})'
            rq1_table.loc[(strategy, case), "f1-score"] = f'{model} ({np.round(f1_score_value, 2)})'

            rq1_table.loc[(strategy, case), "FP"] = f'{model} ({fp})'
            rq1_table.loc[(strategy, case), "FN"] = f'{model} ({fn})'
            rq1_table.loc[(strategy, case), "Allucination Rate"] = f'{model} ({np.round(errors_df.loc[(errors_df['model'] == model) & (errors_df['strategy'] == strategy), 'pct_error'].values[0], 2)})'
    
    rq1_table.to_excel(r'C:\Users\mario\Documents\estudos\ufc\Granularidade Grossa\results\rq1_table.xlsx')
        

get_rq1_table(r'C:\Users\mario\Documents\estudos\ufc\Granularidade Grossa\results\compact_result_table_without_duplicates.xlsx')