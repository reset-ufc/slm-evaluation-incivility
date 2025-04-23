import os
import pandas as pd
from pathlib import Path

results_path = Path('results')
compact_table_path_uncivil = results_path / 'compact_result_table_uncivil_without_duplicates.xlsx'
compact_table_path_civil = results_path / 'compact_result_table_civil_without_duplicates.xlsx'

uncivil = pd.read_excel(compact_table_path_uncivil)
uncivil = uncivil.ffill()
uncivil = uncivil.loc[~uncivil['Model'].isin(['toxicr', 'refined_model']).values]
uncivil = uncivil.rename(columns={'precision': 'Precision', 'recall': 'Recall', 'f1-score': 'F1-score', 'Model': 'Modelo'})
uncivil_sem_combinacoes = uncivil.loc[~uncivil['Strategy'].str.contains('role_based_')].copy()  # excluir combinações com role-based

civil = pd.read_excel(compact_table_path_civil)
civil = civil.ffill()
civil = civil.loc[~civil['Model'].isin(['toxicr', 'refined_model']).values]
civil = civil.rename(columns={'precision': 'Precision', 'recall': 'Recall', 'f1-score': 'F1-score', 'Model': 'Modelo'})
civil_sem_combinacoes = civil.loc[~civil['Strategy'].str.contains('role_based_')].copy()

uncivil_sem_combinacoes["Class"] = "Uncivil"
civil_sem_combinacoes["Class"] = "Civil"

df_geral = pd.concat([uncivil_sem_combinacoes, civil_sem_combinacoes])

df_geral['Precision'] = df_geral['Precision'].round(2)
df_geral['Recall'] = df_geral['Recall'].round(2)

categorias = [
    "Civil", "Uncivil"
]

def read_xlsx_files_from_folder(folder_path):
    xlsx_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    dataframes = {}

    civil_path = results_path / "civil_ml_metric_means_results.xlsx"
    uncivil_path = results_path / "uncivil_ml_metric_means_results.xlsx"
    civil_df = pd.read_excel(civil_path)   
    uncivil_df = pd.read_excel(uncivil_path)

    dataframes["civil_ml_metric_means_results.xlsx"] = [civil_df, "Civil"]
    dataframes["uncivil_ml_metric_means_results.xlsx"] = [uncivil_df, "Uncivil"]

    return dataframes

# Exemplo de uso
folder_path = 'results'
excel_data = read_xlsx_files_from_folder(folder_path)

for file_name, vector in excel_data.items():
    print(f"Contents of {file_name}:")
    print(f"Incivility of this file: {vector[1]}")
    print(vector[0].head())

import pandas as pd
modelos = [
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

subcolunas = ["Pr-diff", "Re-diff", "F1-diff"]

# Build MultiIndex from columns (tuples with (category, subcolumn))
colunas = pd.MultiIndex.from_product([categorias, subcolunas])

# Create empty DataFrame with composite columns and models as index
df = pd.DataFrame(index=modelos, columns=colunas)

df.reset_index(inplace=True)
df.rename(columns={'index': 'Model'}, inplace=True)

print(df.head())

best_model = 'gpt-4o-mini + role_based'
print(best_model)


results_concat = df_geral.copy()
print(results_concat.head())
modelos = results_concat['Modelo'].unique()
estrategias = results_concat['Strategy'].unique()
print(modelos)
modelo_slm = next((x for x in modelos if x in best_model),None)
estrategia = next((x for x in estrategias if x in best_model),None)
print(f"modelo slm:" + modelo_slm + "\nestrategia: " + estrategia)
df_metrics_slm = results_concat[
    (results_concat['Modelo'] == modelo_slm) & 
    (results_concat['Strategy'] == estrategia)
]
print(df_metrics_slm.head())
metricas = ['Precision', 'Recall','F1-score']
mapa_nomes_modelos = {
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


def calcular_diff():
    for file_name, (dataframe_ml, categoria) in excel_data.items():
        if categoria is None:
            continue

        linha = df_metrics_slm[df_metrics_slm['Class'] == categoria]

        if linha.empty:
            print(f"Nenhum dado encontrado para categoria '{categoria}'")
            continue

        for _, row in dataframe_ml.iterrows():
            nome_abreviado = row['Unnamed: 0']
            model_ml = mapa_nomes_modelos.get(nome_abreviado)

            if model_ml not in df['Model'].values:
                print(f"Modelo '{nome_abreviado}' não reconhecido.")
                continue

            pr_diff = linha['Precision'].values[0] - row['Precision']
            re_diff = linha['Recall'].values[0] - row['Recall'] 
            f1_diff = linha['F1-score'].values[0] - row['F1']

            df.loc[df['Model'] == model_ml, (categoria, "Pr-diff")] = pr_diff
            df.loc[df['Model'] == model_ml, (categoria, "Re-diff")] = re_diff
            df.loc[df['Model'] == model_ml, (categoria, "F1-diff")] = f1_diff

calcular_diff()
print(df.head())
df.to_excel(results_path / "rq3_table.xlsx", index=True)
        
