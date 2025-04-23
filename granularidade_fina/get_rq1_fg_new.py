import pandas as pd
import pathlib
from pathlib import Path
# Carrega o CSV com os resultados
path = Path('results_table') / 'results_concat.csv'
if not path.exists():
    raise FileNotFoundError(f"File not found: {path}")
df = pd.read_csv(path)

# Ordem desejada
incivilities = [
    "Bitter Frustration", "Impatience", "Vulgarity", "Irony", "Identify Attack/Name Calling",
    "Threat", "Insulting", "Entitlement", "Mocking", "None"
]
models = ['deepseek-14b', 'deepseek-8b', 'gemma2_9b', 'gemma_7b', 'gpt-4o-mini',
 'llama3.1_8b', 'llama3.2_3b', 'mistral-nemo_12b', 'mistral_7b', 'phi4_14b']


df.rename(columns={'Tbdf': 'Type', 'Modelo': 'Model'}, inplace=True)
print(df['Model'].unique())
# Índice
index = models + ["Average"]
multi_col_data = {}

for incivility in incivilities:
    filtered = df[df['Type'] == incivility.lower()]
    grouped = filtered.groupby('Model')[['Precision', 'Recall', 'F1-score']].mean().reindex(models)
    
    # Corrige e adiciona a linha de média corretamente
    average_row = grouped.mean().to_frame().T
    average_row.index = ['Average']
    grouped = pd.concat([grouped, average_row])
    
    grouped = grouped.round(2)
    
    for metric in ['Precision', 'Recall', 'F1-score']:
        multi_col_data[(incivility, metric)] = grouped[metric]

# Cria dataframe com MultiIndex nas colunas
final_df = pd.DataFrame(multi_col_data, index=index)
final_df.columns = pd.MultiIndex.from_tuples(final_df.columns)

# Salva
final_df.to_excel("tabela_resultados_multicol.xlsx")