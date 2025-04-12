import pandas as pd
from pathlib import Path

def read_csv_file():
    df = pd.read_csv(r'granularidade_fina\results_table\results_concat.csv')
    
    bitter_df = df[df['Tbdf'] == 'bitter frustration']
    impatience_df = df[df['Tbdf'] == 'impatience']
    mocking_df = df[df['Tbdf'] == 'mocking']
    vulgarity_df = df[df['Tbdf'] == 'vulgarity']
    insulting_df = df[df['Tbdf'] == 'insulting']
    none_df = df[df['Tbdf'] == 'none']
    entitlement_df = df[df['Tbdf'] == 'entitlement']
    ia_df = df[df['Tbdf'] == 'identify attack/name calling']
    irony_df = df[df['Tbdf'] == 'irony']
    threat_df = df[df['Tbdf'] == 'threat']
    
    filter_class_list = [bitter_df,impatience_df,mocking_df,vulgarity_df,insulting_df,none_df,entitlement_df,ia_df,irony_df,threat_df]
    
    zero_shot = []
    one_shot = []
    few_shot_3 = []
    auto_cot = []
    role_based = []
    list_strategy_names = ['zero_shot', 'one_shot', 'few_shot_3', 'auto_cot', 'role_based']

    list_strategy = [zero_shot,one_shot,few_shot_3,auto_cot,role_based]
    df_list_strategy_class_filter = []
    
    for i, strategy in enumerate(list_strategy_names):
        for tbdf in filter_class_list:
            df_filter_class_strategy = tbdf[tbdf['Strategy'] == strategy]
            list_strategy[i].append(df_filter_class_strategy)

    return list_strategy



lista = read_csv_file()
print(lista)

def switch_better_global(metric_column: str, strategy_name: str, list_strategy: list, tbdf_index: int):
    """
    Seleciona a melhor linha com base na métrica `metric_column` dentro da estratégia e do tbdf específico.
    """
    strategy_dict = {
        'zero_shot': 0,
        'one_shot': 1,
        'few_shot_3': 2,
        'auto_cot': 3,
        'role_based': 4
    }

    strategy_index = strategy_dict.get(strategy_name)
    dfs = list_strategy[strategy_index]  # Pegamos a lista de DataFrames dessa estratégia

    best_value = float('-inf')
    best_row = None

    df = dfs[tbdf_index]  # Pegamos o DataFrame da classe específica dentro da estratégia

    if metric_column in df.columns and not df.empty:
        row = df.loc[df[metric_column].idxmax()]  # Encontramos o melhor F1-score dentro do df já filtrado
        best_value = row[metric_column]
        best_row = row

    if best_row is not None:
        return {
            'Model': best_row['Modelo'],
            'Class': best_row['Tbdf'],
            'Pr': best_row['Precision'],
            'Re': best_row['Recall'],
            metric_column: best_row[metric_column],
            'Accuracy': best_row['Accuracy'],
            'FP': best_row['Fp'],
            'FN': best_row['Fn']
        }
    else:
        return {
            'Model': 'NaN',
            'Class': 'NaN',
            'Pr': 'NaN',
            'Re': 'NaN',
            metric_column: 'NaN',
            'Accuracy': 'NaN',
            'FP': 'NaN',
            'FN': 'NaN'
        }


def switch_worst_global(metric_column: str, strategy_name: str, list_strategy: list, tbdf_index: int):
    """
    Seleciona a pior linha com base na métrica `metric_column` dentro da estratégia e do tbdf específico.
    """
    strategy_dict = {
        'zero_shot': 0,
        'one_shot': 1,
        'few_shot_3': 2,
        'auto_cot': 3,
        'role_based': 4
    }

    strategy_index = strategy_dict.get(strategy_name)
    dfs = list_strategy[strategy_index]  # Pegamos a lista de DataFrames dessa estratégia

    worst_value = float('inf')
    worst_row = None

    df = dfs[tbdf_index]  # Pegamos o DataFrame da classe específica dentro da estratégia

    if metric_column in df.columns and not df.empty:
        row = df.loc[df[metric_column].idxmin()]  # Encontramos o pior F1-score dentro do df já filtrado
        worst_value = row[metric_column]
        worst_row = row

    if worst_row is not None:
        return {
            'Model': worst_row['Modelo'],
            'Class': worst_row['Tbdf'],
            'Pr': worst_row['Precision'],
            'Re': worst_row['Recall'],
            metric_column: worst_row[metric_column],
            'Accuracy': worst_row['Accuracy'],
            'FP': worst_row['Fp'],
            'FN': worst_row['Fn']
        }
    else:
        return {
            'Model': 'NaN',
            'Class': 'NaN',
            'Pr': 'NaN',
            'Re': 'NaN',
            metric_column: 'NaN',
            'Accuracy': 'NaN',
            'FP': 'NaN',
            'FN': 'NaN'
        }



def fill_table(df_table, list_df, list_strategy, list_tbdf):
    """
    Preenche a tabela `df_table` com os melhores e piores modelos em cada estratégia e tbdf.
    """
    for tbdf_index, tbdf in enumerate(list_tbdf):
        for strategy in list_strategy:
            better = switch_better_global('F1-score', strategy, list_df, tbdf_index)
            df_table.loc[(strategy, "Best"), (tbdf, "Model")] = better['Model']
            df_table.loc[(strategy, "Best"), (tbdf, "Pr")] = better['Pr']
            df_table.loc[(strategy, "Best"), (tbdf, "Re")] = better['Re']
            df_table.loc[(strategy, "Best"), (tbdf, "Accuracy")] = better['Accuracy']
            df_table.loc[(strategy, "Best"), (tbdf, "F1")] = better['F1-score']
            df_table.loc[(strategy, "Best"), (tbdf, "FP")] = better['FP']
            df_table.loc[(strategy, "Best"), (tbdf, "FN")] = better['FN']

            worse = switch_worst_global('F1-score', strategy, list_df, tbdf_index)
            df_table.loc[(strategy, "Worst"), (tbdf, "Model")] = worse['Model']
            df_table.loc[(strategy, "Worst"), (tbdf, "Pr")] = worse['Pr']
            df_table.loc[(strategy, "Worst"), (tbdf, "Re")] = worse['Re']
            df_table.loc[(strategy, "Worst"), (tbdf, "Accuracy")] = worse['Accuracy']
            df_table.loc[(strategy, "Worst"), (tbdf, "F1")] = worse['F1-score']
            df_table.loc[(strategy, "Worst"), (tbdf, "FP")] = worse['FP']
            df_table.loc[(strategy, "Worst"), (tbdf, "FN")] = worse['FN']




        
# Definindo as categorias e as métricas
tbdf_classes = [
    "Bitter Frustration", "Impatience", "Mocking", "Vulgarity","Insulting","None",
    "Entitlement","Identify Attack/Name Calling","Irony", "Threat"
]

metrics = ["Model", "Pr", "Re","Accuracy", "F1", "FP", "FN"]

# Criando o MultiIndex para as colunas
multi_columns = []
for label in tbdf_classes:
    for metric in metrics:
        multi_columns.append((label, metric))

# Criando o MultiIndex com Pandas
column_index = pd.MultiIndex.from_tuples(multi_columns)

# Linhas da tabela (com Strategy e Case)
strategies = ["zero_shot", "one_shot","few_shot_3", "auto_cot", "role_based"]
cases = ["Best", "Worst"]

rows = []
index = []
for strategy in strategies:
    for case in cases:
        index.append((strategy, case))

# Criando o índice composto
index = pd.MultiIndex.from_tuples(index, names=["Strategy", "Case"])

# Criando o DataFrame vazio 
df = pd.DataFrame("", index=index, columns=column_index)
list_strategy_names = ['zero_shot','one_shot','few_shot_3','auto_cot','role_based']   
fill_table(df,lista,list_strategy_names,tbdf_classes)





# Exibindo a tabela
print(df)
df.to_excel(r"granularidade_fina\results_table\rq1.xlsx", merge_cells=True)


