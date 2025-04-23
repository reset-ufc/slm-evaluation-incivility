from pathlib import Path
import pandas as pd
import os
import re

strategies = ['zero_shot','one_shot','few_shot_3', 'role_based', 'auto_cot','role_based_one_shot','role_based_few_shot_3','role_based_auto_cot']
total = 5959

for strategy in strategies:
    # Define the path to the directory containing the CSV files
    results_path = Path('./results/')
    strategy_path = results_path / strategy
    
    models = [Path(m) for m in os.listdir(strategy_path)]

    for model in models:
        # Define the path to the model directory
        model_path = strategy_path / model
        model_name = model.name

        if model_name == 'phi3_3.8b':
            continue

        print(model_name)

        sanitized_model_name = re.sub(r'[^\w\-_]', '_', model_name)
        
        if 'deepseek' in model_name and '14b' in model_name:
            sanitized_model_name = 'deepseek-r1_14b'
        if 'deepseek' in model_name and '8b' in model_name:
            sanitized_model_name = 'deepseek-r1_8b'

        pred_df = pd.read_csv(model_path / f'fineclassify_{sanitized_model_name}_{strategy}.csv')
        errors_idx = pred_df[pred_df['Tbdf'] == 'error'].index.tolist()
        timeouts_idx = pred_df[pred_df['Tbdf'] == 'timeout'].index.tolist()

        if len(errors_idx) > 0:
            pred_df = pred_df.drop(errors_idx)
        if len(timeouts_idx) > 0:
            pred_df = pred_df.drop(timeouts_idx)
        
        pred_df.to_csv(model_path / f'fineclassify_{sanitized_model_name}_{strategy}.csv', index=False)
        

