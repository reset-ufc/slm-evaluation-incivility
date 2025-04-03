import pandas as pd
from pathlib import Path

def get_quantity_errors():
    results_path = Path("results")
    duplicates = pd.read_csv(r"C:\Users\mario\Documents\estudos\ufc\LLM-TESTS\granularidade_grossa\data\all_duplicates.csv", index_col=0)
    models_path = [path for path in results_path.iterdir() if path.is_dir() and path.name not in ["deepseek-r1_14b", 'refined_model', 'toxicr']]
    df_count_errors = pd.DataFrame(columns=["model", "strategy", "total_errors"])

    for model in models_path:
        strategy_path = [path for path in model.iterdir() if path.is_dir()]
        for strategy in strategy_path:
            error_path = strategy / "errors.json"
            error_df = pd.read_json(error_path)

            if len(error_df) != 0:
                error_df = error_df.loc[~error_df['index'].isin(duplicates.index.to_list())] # removendo os duplicados
            count_errors = len(error_df)
            df_count_errors.loc[len(df_count_errors)] = [model.name, strategy.name, count_errors]
    df_count_errors['total'] = 6879
    df_count_errors['classified_samples'] = df_count_errors['total'] - df_count_errors['total_errors']
    df_count_errors['pct_error'] = df_count_errors['total_errors'] / df_count_errors['classified_samples']
    return df_count_errors