import pandas as pd 
import numpy as np
from get_quantity_errors import get_quantity_errors
from pathlib import Path

def get_rq2_table(rq1_table_path, compact_table_path):
    rq1_table_uncivil = pd.read_excel(rq1_table_path, sheet_name='UNCIVIL', header=2)
    rq1_table_uncivil.columns = ['Strategy', 'Case', 'Model', 'precision', 'recall', 'f1-score', "acuraccy", "roc_auc", 'FP', 'FN']
    rq1_table_uncivil = rq1_table_uncivil.ffill()
    rq1_table_uncivil['Strategy'] = rq1_table_uncivil['Strategy'].ffill()

    rq1_table_civil = pd.read_excel(rq1_table_path, sheet_name='CIVIL', header=2)
    rq1_table_civil.columns = ['Strategy', 'Case', 'Model', 'precision', 'recall', 'f1-score', "acuraccy", "roc_auc", 'FP', 'FN']
    rq1_table_civil = rq1_table_civil.ffill()
    rq1_table_civil['Strategy'] = rq1_table_civil['Strategy'].ffill()

    rq1_table = pd.concat([rq1_table_uncivil, rq1_table_civil], axis=0)

    # print(rq1_table)

    compact_table = pd.read_excel(compact_table_path)
    compact_table = compact_table.ffill()
    # errors_df = get_quantity_errors()

    strategies = ['zero_shot', "one_shot", 'few_shot', 'auto_cot', 'role_based']
    combination_sets = ["Raw", "Role-based"]
    columns = ["precision", "recall", "f1-score", "accuracy", "roc_auc", "FP", "FN"]

    # extract worst models
    worst_models = rq1_table.loc[rq1_table.isin(strategies).any(axis=1), ['f1-score', 'Case', 'Model']]
    worst_models = worst_models.loc[worst_models['Case'] == 'Worst']['Model'].values.tolist()

    strategies.remove('role_based')

    unique_worst_models = list(set(worst_models))

    print(unique_worst_models)

    multi_index = pd.MultiIndex.from_product(
    [unique_worst_models, strategies, combination_sets], 
    names=["Model", "Strategy", "Combination set"]
    )

    rq2_table = pd.DataFrame(index=multi_index, columns=columns)

    for model in unique_worst_models:
        for strategy in strategies:
            for combination_set in combination_sets:
                if combination_set == "Raw":
                    model_path = compact_table.loc[(compact_table['Model'] == model) & (compact_table['Strategy'] == strategy)]
                else:
                    model_path = compact_table.loc[(compact_table['Model'] == model) & (compact_table['Strategy'] == f'role_based_{strategy}')]

                    if strategy == 'zero_shot':
                        model_path = compact_table.loc[(compact_table['Model'] == model) & (compact_table['Strategy'] == 'role_based')]

                f1_score_value = model_path['f1-score'].values[0]
                precision_value = model_path['precision'].values[0]
                recall_value = model_path['recall'].values[0]
                fp = model_path['FP'].values[0]
                fn = model_path['FN'].values[0]
                accuracy_value = model_path['accuracy'].values[0]
                roc_auc_value = model_path['roc_auc'].values[0]

                # rq2_table.sort_index(inplace=True)

                rq2_table.loc[(model, strategy, combination_set), "precision"] = np.round(precision_value, 2)
                rq2_table.loc[(model, strategy, combination_set), "recall"] = np.round(recall_value, 2)
                rq2_table.loc[(model, strategy, combination_set), "f1-score"] = np.round(f1_score_value, 2)
                rq2_table.loc[(model, strategy, combination_set), "accuracy"] = np.round(accuracy_value, 2)
                rq2_table.loc[(model, strategy, combination_set), "roc_auc"] = np.round(roc_auc_value, 2)

                rq2_table.loc[(model, strategy, combination_set), "FP"] = fp
                rq2_table.loc[(model, strategy, combination_set), "FN"] = fn

        
    results_path = Path('results')    
    rq2_table.to_excel(results_path / 'rq2_table_worst.xlsx')

results_path = Path('results')
rq1_table_path = results_path / 'new_rq1_table.xlsx'
compact_table_path = results_path / 'compact_result_table_uncivil_without_duplicates.xlsx'

get_rq2_table(rq1_table_path, compact_table_path)
