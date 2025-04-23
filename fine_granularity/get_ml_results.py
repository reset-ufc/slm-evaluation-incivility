import pandas as pd
import numpy as np
from pathlib import Path
import json

# Main paths
results_path = Path('results')
ml_results_path = results_path / 'ml_models'
folds_path = ml_results_path / 'folds'

models = ['ADA', 'LRC', 'MNB', 'RFC']
representations = ['BoW', 'TF-IDF']

encoder = {'none': 0, 'entitlement': 1, 'impatience': 2, 'bitter frustration': 3, 'mocking': 4, 'irony': 5, 'vulgarity': 6, 'identify attack/name calling': 7, 'insulting': 8, 'threat': 9}

# Dict to store results
results = {
    f"{model}_{rep}": {'none': {'precision': [], 'recall': [], 'f1': []},
                       'entitlement': {'precision': [], 'recall': [], 'f1': []},
                       'impatience': {'precision': [], 'recall': [], 'f1': []},
                       'bitter frustration': {'precision': [], 'recall': [], 'f1': []},
                       'mocking': {'precision': [], 'recall': [], 'f1': []},
                       'irony': {'precision': [], 'recall': [], 'f1': []},
                       'vulgarity': {'precision': [], 'recall': [], 'f1': []},
                       'identify attack/name calling': {'precision': [], 'recall': [], 'f1': []},
                       'insulting': {'precision': [], 'recall': [], 'f1': []},
                       'threat': {'precision': [], 'recall': [], 'f1': []}
                       }
    for model in models for rep in representations
}

# Iterate to each fold and representation
for fold_dir in folds_path.iterdir():
    for rep in representations:
        rep_path = fold_dir / rep
        if not rep_path.exists():
            continue
        for report_file in rep_path.iterdir():
            if report_file.suffix != '.csv':
                continue

            report_df = pd.read_csv(report_file)
            precision_none, recall_none, f1_none = report_df.loc[0, ['precision', 'recall', 'f1-score']]
            precision_entitlement, recall_entitlement, f1_entitlement = report_df.loc[1, ['precision', 'recall', 'f1-score']]
            precision_impatience, recall_impatience, f1_impatience = report_df.loc[2, ['precision', 'recall', 'f1-score']]
            precision_bitter, recall_bitter, f1_bitter = report_df.loc[3, ['precision', 'recall', 'f1-score']]
            precision_mocking, recall_mocking, f1_mocking = report_df.loc[4, ['precision', 'recall', 'f1-score']]
            precision_irony, recall_irony, f1_irony = report_df.loc[5, ['precision', 'recall', 'f1-score']]
            precision_vulgarity, recall_vulgarity, f1_vulgarity = report_df.loc[6, ['precision', 'recall', 'f1-score']]
            precision_ia, recall_ia, f1_ia = report_df.loc[7, ['precision', 'recall', 'f1-score']]
            precision_insulting, recall_insulting, f1_insulting = report_df.loc[8, ['precision', 'recall', 'f1-score']]
            precision_threat, recall_threat, f1_threat = report_df.loc[9, ['precision', 'recall', 'f1-score']]

            for model in models:
                if model in report_file.name:
                    key = f"{model}_{rep}"
                    results[key]['none']['precision'].append(precision_none)
                    results[key]['none']['recall'].append(recall_none)
                    results[key]['none']['f1'].append(f1_none)

                    results[key]['entitlement']['precision'].append(precision_entitlement)
                    results[key]['entitlement']['recall'].append(recall_entitlement)
                    results[key]['entitlement']['f1'].append(f1_entitlement)

                    results[key]['impatience']['precision'].append(precision_impatience)
                    results[key]['impatience']['recall'].append(recall_impatience)
                    results[key]['impatience']['f1'].append(f1_impatience)

                    results[key]['bitter frustration']['precision'].append(precision_bitter)
                    results[key]['bitter frustration']['recall'].append(recall_bitter)
                    results[key]['bitter frustration']['f1'].append(f1_bitter)

                    results[key]['mocking']['precision'].append(precision_mocking)
                    results[key]['mocking']['recall'].append(recall_mocking)
                    results[key]['mocking']['f1'].append(f1_mocking)

                    results[key]['irony']['precision'].append(precision_irony)
                    results[key]['irony']['recall'].append(recall_irony)
                    results[key]['irony']['f1'].append(f1_irony)

                    results[key]['vulgarity']['precision'].append(precision_vulgarity)
                    results[key]['vulgarity']['recall'].append(recall_vulgarity)
                    results[key]['vulgarity']['f1'].append(f1_vulgarity)

                    results[key]['identify attack/name calling']['precision'].append(precision_ia)
                    results[key]['identify attack/name calling']['recall'].append(recall_ia)
                    results[key]['identify attack/name calling']['f1'].append(f1_ia)

                    results[key]['insulting']['precision'].append(precision_insulting)
                    results[key]['insulting']['recall'].append(recall_insulting)
                    results[key]['insulting']['f1'].append(f1_insulting)

                    results[key]['threat']['precision'].append(precision_threat)
                    results[key]['threat']['recall'].append(recall_threat)
                    results[key]['threat']['f1'].append(f1_threat)
                    break

# Calculate the mean for each class and model
mean_results = {k: {
    'none': {metric: np.mean(v['none'][metric]) for metric in ['precision', 'recall', 'f1']},
    'entitlement': {metric: np.mean(v['entitlement'][metric]) for metric in ['precision', 'recall', 'f1']},
    'impatience': {metric: np.mean(v['impatience'][metric]) for metric in ['precision', 'recall', 'f1']},
    'bitter frustration': {metric: np.mean(v['bitter frustration'][metric]) for metric in ['precision', 'recall', 'f1']},
    'mocking': {metric: np.mean(v['mocking'][metric]) for metric in ['precision', 'recall', 'f1']},
    'irony': {metric: np.mean(v['irony'][metric]) for metric in ['precision', 'recall', 'f1']},
    'vulgarity': {metric: np.mean(v['vulgarity'][metric]) for metric in ['precision', 'recall', 'f1']},
    'identify attack/name calling': {metric: np.mean(v['identify attack/name calling'][metric]) for metric in ['precision', 'recall', 'f1']},
    'insulting': {metric: np.mean(v['insulting'][metric]) for metric in ['precision', 'recall', 'f1']},
    'threat': {metric: np.mean(v['threat'][metric]) for metric in ['precision', 'recall', 'f1']}
} for k, v in results.items()}

# DistilBERT Results
bert_results_path = results_path / 'distil_bert_results'
with open(bert_results_path / 'DistilBert-10folds_results.json', 'r') as f:
    bert_data = json.load(f)

bert_metrics = {'none': {'precision': 0, 'recall': 0, 'f1': 0},
                'entitlement': {'precision': 0, 'recall': 0, 'f1': 0},
                'impatience': {'precision': 0, 'recall': 0, 'f1': 0},
                'bitter frustration': {'precision': 0, 'recall': 0, 'f1': 0},
                'mocking': {'precision': 0, 'recall': 0, 'f1': 0},
                'irony': {'precision': 0, 'recall': 0, 'f1': 0},
                'vulgarity': {'precision': 0, 'recall': 0, 'f1': 0},
                'identify attack/name calling': {'precision': 0, 'recall': 0, 'f1': 0},
                'insulting': {'precision': 0, 'recall': 0, 'f1': 0},
                'threat': {'precision': 0, 'recall': 0, 'f1': 0}
                }


for fold_metrics in bert_data.values():
    for cls, name_class in enumerate(list(bert_metrics.keys())):
        bert_metrics[name_class]['precision'] += fold_metrics[f'pre_cls_{str(cls)}']
        bert_metrics[name_class]['recall'] += fold_metrics[f'rec_cls_{str(cls)}']
        bert_metrics[name_class]['f1'] += fold_metrics[f'f1_cls_{str(cls)}']

num_folds = len(bert_data)
for label in list(bert_metrics.keys()):
    for metric in ['precision', 'recall', 'f1']:
        bert_metrics[label][metric] /= num_folds

mean_results['DistilBERT'] = bert_metrics

# Generate the DataFrames 
def build_df(label):
    rows = []
    indices = []
    for model_name, scores in mean_results.items():
        indices.append(model_name)
        model_scores = scores[label]
        rows.append([model_scores['precision'], model_scores['recall'], model_scores['f1']])
    return pd.DataFrame(rows, index=indices, columns=['Precision', 'Recall', 'F1'])

for label in bert_metrics.keys():
    df = build_df(label)

    df = df.round(2)

    # Salvar os CSVs
    df.to_excel(results_path / f'{label.replace(' ', '_').replace('/', '_')}_ml_metric_means_results.xlsx')
