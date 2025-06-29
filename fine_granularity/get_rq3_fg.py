import os
import pandas as pd
from get_best_model import get_best_model
from pathlib import Path

# Original categories of incivility
categories = [
    "Bitter Frustration", "Impatience", "Vulgarity", "Irony",
    "Identify Attack/Name Calling", "Threat", "Insulting",
    "Entitlement", "Mocking", "None"
]

# Optional renaming for display/export
incivility_rename_map = {
    "Identify Attack/Name Calling": "Identify Attack"
}

def read_xlsx_files_from_folder(folder_path):
    # xlsx_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    xlsx_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx') and 'balancing' in f]
    dataframes = {}

    for file in xlsx_files:
        file_path = os.path.join(folder_path, file)
        category = next((x for x in categories if x.upper().replace(" ", "_").replace("/", "_") in file.upper()), None)
        dataframe = pd.read_excel(file_path)

        # Round metrics before calculations
        for col in ['Precision', 'Recall', 'F1']:
            if col in dataframe.columns:
                dataframe[col] = dataframe[col].round(2)

        dataframes[file] = [dataframe, category]
        print(f"File: {file} | Detected category: {category}")

    return dataframes

# Load the Excel files
folder_path = Path('results')
excel_data = read_xlsx_files_from_folder(folder_path)

# Preview each file loaded
for file_name, vector in excel_data.items():
    print(f"Contents of {file_name}:")
    print(f"Detected incivility category: {vector[1]}")
    print(vector[0].head())

# List of model names
models = [
    "Adaptive Boosting + BoW",
    "Logistic Regression + BoW",
    "Multinomial Naive Bayes + BoW",
    "Random Forest + BoW",
    "Adaptive Boosting + TF-IDF",
    "Logistic Regression + TF-IDF",
    "Multinomial Naive Bayes + TF-IDF",
    "Random Forest + TF-IDF",
    "DistilBERT"
]

# MultiIndex columns: (category, metric)
subcolumns = ["Pr-diff", "Re-diff", "F1-diff"]
columns = pd.MultiIndex.from_product([categories, subcolumns])
df = pd.DataFrame(index=models, columns=columns)
df.reset_index(inplace=True)
df.rename(columns={'index': 'Model'}, inplace=True)

print(df.head())

# Load the best model (from external module)
best_model = get_best_model()
print(best_model)

# Load full results table
results_table_path = Path('results_table')
results_table_path.mkdir(parents=True, exist_ok=True)
results_concat = pd.read_csv(results_table_path / "results_concat.csv")

# Round result metrics before calculation
for col in ['Precision', 'Recall', 'F1-score']:
    if col in results_concat.columns:
        results_concat[col] = results_concat[col].round(2)

# Identify best model + strategy combination
model_list = results_concat['Modelo'].unique()
strategy_list = results_concat['Strategy'].unique()
selected_model = next((x for x in model_list if x in best_model), None)
selected_strategy = next((x for x in strategy_list if x in best_model), None)
print(f"Selected model: {selected_model}\nStrategy: {selected_strategy}")

# Filter results for the best model/strategy
df_metrics_slm = results_concat[
    (results_concat['Modelo'] == selected_model) & 
    (results_concat['Strategy'] == selected_strategy)
]

print(df_metrics_slm.head())

# Mapping short model names to full names
model_name_map = {
    'ADA_BoW': "Adaptive Boosting + BoW",
    'LRC_BoW': "Logistic Regression + BoW",
    'MNB_BoW': "Multinomial Naive Bayes + BoW",
    'RFC_BoW': "Random Forest + BoW",
    'ADA_TF-IDF': "Adaptive Boosting + TF-IDF",
    'LRC_TF-IDF': "Logistic Regression + TF-IDF",
    'MNB_TF-IDF': "Multinomial Naive Bayes + TF-IDF",
    'RFC_TF-IDF': "Random Forest + TF-IDF",
    'DistilBERT': "DistilBERT"
}

# Difference calculation function
def calculate_diff():
    for file_name, (df_model, category) in excel_data.items():
        if category is None:
            continue
        category_normalized = category.lower()
        row = df_metrics_slm[df_metrics_slm['Tbdf'] == category_normalized]
        if row.empty:
            print(f"No data found for category '{category_normalized}'")
            continue
        for _, model_row in df_model.iterrows():
            short_name = model_row['Unnamed: 0']
            full_name = model_name_map.get(short_name)
            if full_name not in df['Model'].values:
                print(f"Model '{short_name}' not recognized.")
                continue
            pr_diff = round(row['Precision'].values[0] - model_row['Precision'], 2)
            re_diff = round(row['Recall'].values[0] - model_row['Recall'], 2)
            f1_diff = round(row['F1-score'].values[0] - model_row['F1'], 2)
            df.loc[df['Model'] == full_name, (category, "Pr-diff")] = pr_diff
            df.loc[df['Model'] == full_name, (category, "Re-diff")] = re_diff
            df.loc[df['Model'] == full_name, (category, "F1-diff")] = f1_diff

# Run difference calculation
calculate_diff()
print(df.head())

# Prepare for export: rename categories for display
df_export = df.copy()
df_export.columns = pd.MultiIndex.from_tuples([
    (incivility_rename_map.get(cat, cat), metric) for cat, metric in df.columns
])
df_export.set_index("Model", inplace=True)

# Save to Excel
path = results_table_path / 'rq3_balancing.xlsx'
df_export.to_excel(path)
