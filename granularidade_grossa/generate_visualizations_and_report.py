from prompts import prompt_factory, strategies
import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os

cur_dir = Path(os.getcwd())
data_dir = cur_dir / 'data' / 'final_df.csv'
results_dir = cur_dir / 'results'

models = ['gemma:7b', 'gemma2:9b', 'mistral-nemo:12b', 'mistral:7b', 'deepseek-r1:8b', 'deepseek-r1:14b', 'llama3.2:3b', 'llama3.1:8b', 'gpt-4o-mini']

for model in models:
    model_path = results_dir / model.replace(':', '_')
    for strategy in strategies:
        # load the predictions csv
        strategy_path = model_path / strategy
        predictions_csv = strategy_path / 'predictions_df.csv'

        data = pd.read_csv(data_dir)
        predictions_df = pd.read_csv(predictions_csv)

        truth = predictions_df['actual']
        pred = predictions_df['prediction']

        # Generate classification report
        report = classification_report(truth, pred, output_dict=True)

        report_df = pd.DataFrame(report).transpose()
        result_csv_path = strategy_path / f'report_{model.replace(":", "_")}_{strategy}_2.csv'
        report_df.to_csv(result_csv_path, index=True)

        print(f"Classification report saved to {result_csv_path}")

        # Generate confusion matrix
        cm = confusion_matrix(truth, pred)
        cm_df = pd.DataFrame(cm, index=['CIVIL', 'UNCIVIL'], columns=['CIVIL', 'UNCIVIL'])
        cm_csv = strategy_path / 'confusion_matrix.csv'
        cm_df.to_csv(strategy_path / f'cm_df_{model.replace(":", "_")}_{strategy}_2.csv')

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=['CIVIL', 'UNCIVIL'], yticklabels=['CIVIL', 'UNCIVIL'])
        plt.title(f'Confusion Matrix of {model} - {strategy}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        heatmap_path = strategy_path / f'cm_{model.replace(":", "_")}_{strategy}_2.png'
        plt.savefig(str(heatmap_path), dpi=300)
        plt.close()