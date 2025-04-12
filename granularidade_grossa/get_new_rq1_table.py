import pandas as pd 
import numpy as np
from pathlib import Path

def get_rq1_table():

    # Definindo MultiIndex
    cases = ["Best", "Worst"]
    strategies = ["zero_shot", "one_shot", "few_shot", "auto_cot", "role_based"]
    categories = ["CIVIL", "UNCIVIL"]
    tables = []

    for category in categories:
        results_path = Path(r"results")
        table_name = f"compact_result_table_{category.lower()}_without_duplicates.xlsx"
        compact_table_path = results_path / table_name
        compact_table = pd.read_excel(compact_table_path)
        compact_table = compact_table.ffill()

        # Criando MultiIndex

        index = pd.MultiIndex.from_product(
            [
                strategies,  # Strategy
                cases  # Case
            ],
            names=["Strategy", "Case"]
        )

        # Criando DataFrame
        columns = pd.MultiIndex.from_tuples(
            [(category, "Model"), (category, "precision"), (category, "recall"), (category, "f1-score"), (category, "accuracy"), (category, "roc_auc"), (category, "FP"), (category, "FN")]
        )

        df = pd.DataFrame(
            index=index,
            columns=columns
        )

        best_models_by_strategy = (compact_table.loc[compact_table.groupby('Strategy')['f1-score'].idxmax(), ['Strategy', 'Model', 'precision', 'recall', 'f1-score', 'FP', 'FN']]
                                .sort_values(by='f1-score', ascending=False))


        worst_models_by_strategy = (compact_table.loc[compact_table.groupby('Strategy')['f1-score'].idxmin(), ['Strategy', 'Model', 'precision', 'recall', 'f1-score', 'FP', 'FN']]
                                .sort_values(by='f1-score', ascending=False))

        print(best_models_by_strategy)
        for strategy in strategies:
            for case in cases:
                #print(strategy, case)
                if case == 'Best':
                    #print(best_models_by_strategy.loc[best_models_by_strategy['Strategy'] == strategy.lower(), 'Model'])
                    model = best_models_by_strategy.loc[best_models_by_strategy['Strategy'] == strategy.lower(), 'Model'].values[0]
                    
                else:
                    model = worst_models_by_strategy.loc[worst_models_by_strategy['Strategy'] == strategy.lower(), 'Model'].values[0]
                # print(strategy, case, model)
                
                model_path = compact_table.loc[(compact_table['Strategy'] == strategy) & (compact_table['Model'] == model)]
                f1_score_value = model_path['f1-score'].values[0]
                precision_value = model_path['precision'].values[0]
                recall_value = model_path['recall'].values[0]
                accuracy_value = model_path['accuracy'].values[0]
                roc_auc_value = model_path['roc_auc'].values[0]
                fp = model_path['FP'].values[0]
                fn = model_path['FN'].values[0]

                df.loc[(strategy, case), (category, "Model")] = model
                df.loc[(strategy, case), (category, "precision")] = np.round(precision_value, 2)
                df.loc[(strategy, case), (category, "recall")] = np.round(recall_value, 2)
                df.loc[(strategy, case), (category, "f1-score")] = np.round(f1_score_value, 2)
                df.loc[(strategy, case), (category, "accuracy")] = accuracy_value
                df.loc[(strategy, case), (category, "roc_auc")] = roc_auc_value
                df.loc[(strategy, case), (category, "FP")] = fp
                df.loc[(strategy, case), (category, "FN")] = fn
        
        tables.append(df)
    
    with pd.ExcelWriter(results_path / 'new_rq1_table.xlsx') as writer:
        for i, table in enumerate(tables):
            table.to_excel(writer, sheet_name=categories[i])

get_rq1_table()
