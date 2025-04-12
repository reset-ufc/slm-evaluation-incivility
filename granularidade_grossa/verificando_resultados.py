from pathlib import Path
import pandas as pd
import os

models = ['gemma:7b', 'gemma2:9b', 'mistral-nemo:12b', 'mistral:7b', 'deepseek-r1:8b', 'deepseek-r1:14b', 'llama3.2:3b', 'llama3.1:8b', 'gpt-4o-mini', "phi4_14b"]

curr_path = Path(os.getcwd())
results_path = curr_path / 'results'

for model in models:
    model_path = results_path / model.replace(':', '_')

    strategies = [Path(s) for s in os.listdir(model_path)]

    for strategy in strategies:
 
        predictions_path = model_path / strategy / 'predictions_df.csv'

        if not predictions_path.exists():
            print(f"Prediction not found for {str(model)} on {str(strategy)}")
            continue

        df = pd.read_csv(predictions_path)
        print(f"Number of predictions of {str(model)} on {str(strategy)}:", len(df))

