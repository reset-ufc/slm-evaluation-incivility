import pandas as pd
import numpy as np
from pathlib import Path
import json

# Caminhos principais
results_path = Path('results')
ml_results_path = results_path / 'ml_models_tuning_balancing'
folds_path = ml_results_path / 'folds'

# Modelos que vamos buscar
models = ['ADA', 'LRC', 'MNB', 'RFC']
representations = ['BoW', 'TF-IDF']

# Dicionário para armazenar resultados por modelo e representação
results = {
    f"{model}_{rep}": {'civil': {'precision': [], 'recall': [], 'f1': []},
                       'uncivil': {'precision': [], 'recall': [], 'f1': []}}
    for model in models for rep in representations
}

# Itera por cada fold
for fold_dir in folds_path.iterdir():
    for rep in representations:
        rep_path = fold_dir / rep
        if not rep_path.exists():
            continue
        for report_file in rep_path.iterdir():
            if report_file.suffix != '.csv':
                continue

            report_df = pd.read_csv(report_file)
            precision_civil, recall_civil, f1_civil = report_df.loc[0, ['precision', 'recall', 'f1-score']]
            precision_uncivil, recall_uncivil, f1_uncivil = report_df.loc[1, ['precision', 'recall', 'f1-score']]

            for model in models:
                if model in report_file.name:
                    key = f"{model}_{rep}"
                    results[key]['civil']['precision'].append(precision_civil)
                    results[key]['civil']['recall'].append(recall_civil)
                    results[key]['civil']['f1'].append(f1_civil)
                    results[key]['uncivil']['precision'].append(precision_uncivil)
                    results[key]['uncivil']['recall'].append(recall_uncivil)
                    results[key]['uncivil']['f1'].append(f1_uncivil)
                    break

# Calcular médias
mean_results = {k: {
    'civil': {metric: np.mean(v['civil'][metric]) for metric in ['precision', 'recall', 'f1']},
    'uncivil': {metric: np.mean(v['uncivil'][metric]) for metric in ['precision', 'recall', 'f1']}
} for k, v in results.items()}

# Resultados do DistilBERT
bert_results_path = results_path / 'distil_bert_results'
with open(bert_results_path / 'DistilBert-10folds_results.json', 'r') as f:
    bert_data = json.load(f)

bert_metrics = {'civil': {'precision': 0, 'recall': 0, 'f1': 0},
                'uncivil': {'precision': 0, 'recall': 0, 'f1': 0}}

for fold_metrics in bert_data.values():
    bert_metrics['civil']['precision'] += fold_metrics['pre_0']
    bert_metrics['civil']['recall'] += fold_metrics['rec_0']
    bert_metrics['civil']['f1'] += fold_metrics['f1_0']

    bert_metrics['uncivil']['precision'] += fold_metrics['pre_1']
    bert_metrics['uncivil']['recall'] += fold_metrics['rec_1']
    bert_metrics['uncivil']['f1'] += fold_metrics['f1_1']

num_folds = len(bert_data)
for label in ['civil', 'uncivil']:
    for metric in ['precision', 'recall', 'f1']:
        bert_metrics[label][metric] /= num_folds

# Adiciona ao dicionário de médias
mean_results['DistilBERT'] = bert_metrics

# Geração dos DataFrames finais
def build_df(label):
    rows = []
    indices = []
    for model_name, scores in mean_results.items():
        indices.append(model_name)
        model_scores = scores[label]
        rows.append([model_scores['precision'], model_scores['recall'], model_scores['f1']])
    return pd.DataFrame(rows, index=indices, columns=['Precision', 'Recall', 'F1'])

df_civil = build_df('civil')
df_uncivil = build_df('uncivil')

df_civil = df_civil.round(2)
df_uncivil = df_uncivil.round(2)

# Salvar os dados
df_civil.to_excel(results_path / 'civil_ml_metric_means_results_tuning_balancing.xlsx')
df_uncivil.to_excel(results_path / 'uncivil_ml_metric_means_results_tuning_balancing.xlsx')
