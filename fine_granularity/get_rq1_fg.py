import pandas as pd
import pathlib
from pathlib import Path
# Load the csv with the results
path = Path('results_table') / 'results_concat.csv'
if not path.exists():
    raise FileNotFoundError(f"File not found: {path}")
df = pd.read_csv(path)

# right order
incivilities = [
    "Bitter Frustration", "Impatience", "Vulgarity", "Irony", "Identify Attack/Name Calling",
    "Threat", "Insulting", "Entitlement", "Mocking", "None"
]
models = ['deepseek-14b', 'deepseek-8b', 'gemma2_9b', 'gemma_7b', 'gpt-4o-mini',
 'llama3.1_8b', 'llama3.2_3b', 'mistral-nemo_12b', 'mistral_7b', 'phi4_14b']

incivility_rename_map = {
    "Identify Attack/Name Calling": "Identify Attack",
    # maintain the others
}
model_name_map = {
    'gemma_7b': 'gemma:7b',
    'gemma2_9b': 'gemma2:9b',
    'mistral-nemo_12b': 'mistral-nemo:12b',
    'mistral_7b': 'mistral:7b',
    'deepseek-8b': 'deepseek-r1:8b',
    'deepseek-14b': 'deepseek-r1:14b',
    'llama3.2_3b': 'llama3.2:3b',
    'llama3.1_8b': 'llama3.1:8b',
    'phi4_14b': 'phi4:14b',
    'gpt-4o-mini': 'gpt-4o-mini', # stay the same
    'Average': 'Average'  # stay the same
}

df.rename(columns={'Tbdf': 'Type', 'Modelo': 'Model'}, inplace=True)
print(df['Model'].unique())
# Índice
index = models + ["Average"]
multi_col_data = {}

for incivility in incivilities:
    filtered = df[df['Type'] == incivility.lower()]
    grouped = filtered.groupby('Model')[['Precision', 'Recall', 'F1-score']].mean().reindex(models)
    grouped = grouped.round(2)

    # Calculate the average row
    average_row = grouped.mean().round(2).to_frame().T
    average_row.index = ['Average']

    # Merges the average row with the grouped DataFrame
    grouped = pd.concat([grouped, average_row])
    

    
    for metric in ['Precision', 'Recall', 'F1-score']:
        renamed_incivility = incivility_rename_map.get(incivility, incivility)
        multi_col_data[(renamed_incivility, metric)] = grouped[metric]



# Build the DataFrame with MultiIndex columns
final_df = pd.DataFrame(multi_col_data, index=index)
final_df.columns = pd.MultiIndex.from_tuples(final_df.columns)


# Apply the renaming to the index
final_df.index = final_df.index.map(lambda x: model_name_map.get(x, x))
final_df.index.name = 'Model'  

output_path = Path('results_table') / 'rq1.xlsx'
final_df.to_excel(output_path, index=True, sheet_name='RQ1', engine='openpyxl')
