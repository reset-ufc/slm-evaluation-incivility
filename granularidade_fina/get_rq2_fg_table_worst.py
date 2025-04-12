import pandas as pd

# Carregar dados
df_dados = pd.read_csv(r'granularidade_fina\results_table\results_concat.csv')

# Estratégias e tipos
strategies = {
    "Zero-shot": [
        "Bitter Frustration", "Impatience", "Vulgarity", "Irony", "identify attack/name calling",
        "Threat", "Insulting", "Entitlement", "Mocking", "None"
    ],
    "One-shot": [
        "Bitter Frustration", "Impatience", "Vulgarity", "Irony", "identify attack/name calling",
        "Threat", "Insulting", "Entitlement", "Mocking", "None"
    ],
    "Few-shot": [
        "Bitter Frustration", "Impatience", "Vulgarity", "Irony", "identify attack/name calling",
        "Threat", "Insulting", "Entitlement", "Mocking", "None"
    ],
    "Auto-CoT": [
        "Bitter Frustration", "Impatience", "Vulgarity", "Irony", "identify attack/name calling",
        "Threat", "Insulting", "Mocking", "Entitlement", "None"
    ]
}

# Métricas
metrics = ['Pr', 'Re','Accuracy', 'F1']
categories = ['Without Role-based', 'With Role-based']
multi_cols = pd.MultiIndex.from_product([categories, metrics])

# Preparar estrutura da tabela
data = []
strategy_labels = []
type_labels = []

for strategy, types in strategies.items():
    for t in types:
        strategy_labels.append(strategy)
        type_labels.append(t)
        data.append([None] * len(multi_cols))

df_tabela = pd.DataFrame(data, columns=multi_cols)
df_tabela.insert(0, ('', 'Type'), type_labels)
df_tabela.insert(0, ('', 'Strategy'), strategy_labels)
df_tabela.columns = pd.MultiIndex.from_tuples(df_tabela.columns)
df_tabela[('', 'Type')] = df_tabela[('', 'Type')].str.lower()

# Mapeamento de estratégias
map_strategy = {
    "zero_shot": "Zero-shot",
    "one_shot": "One-shot",
    "few_shot_3": "Few-shot",
    "auto_cot": "Auto-CoT",
    "role_based": "Zero-shot",
    "role_based_one_shot": "One-shot",
    "role_based_few_shot_3": "Few-shot",
    "role_based_auto_cot": "Auto-CoT"
}

# Padronização dos dados
df_dados['Tbdf'] = df_dados['Tbdf'].str.strip().str.lower()
df_dados['Strategy'] = df_dados['Strategy'].str.strip()

# Separar dados base (sem role-based)
estrategias_base = ['zero_shot', 'one_shot', 'few_shot_3', 'auto_cot']
df_base = df_dados[df_dados['Strategy'].isin(estrategias_base)]
idx_piores = df_base.groupby(['Strategy', 'Tbdf'])['F1-score'].idxmin()
df_piores = df_base.loc[idx_piores].reset_index(drop=True)

# Separar dados com role-based
df_role = df_dados[df_dados['Strategy'].str.startswith('role_based')]
idx_role_piores = df_role.groupby(['Strategy', 'Tbdf'])['F1-score'].idxmin()
df_role_piores = df_role.loc[idx_role_piores].reset_index(drop=True)

# Dicionário para lookup rápido dos piores role-based
piores_role = {
    (map_strategy[row['Strategy']], row['Tbdf']): row
    for _, row in df_role_piores.iterrows()
}

# Adicionar coluna para modelo utilizado
df_tabela.insert(2, ('', 'Model Used'), [''] * len(df_tabela))

# Preenchimento da tabela
for _, row in df_piores.iterrows():
    strategy_original = row['Strategy']
    strategy_mapped = map_strategy[strategy_original]
    tipo = row['Tbdf']

    mask = (
        df_tabela[('', 'Strategy')] == strategy_mapped
    ) & (
        df_tabela[('', 'Type')] == tipo
    )

    # Preencher Without Role-based
    df_tabela.loc[mask, ('Without Role-based', 'Pr')] = row['Precision']
    df_tabela.loc[mask, ('Without Role-based', 'Re')] = row['Recall']
    df_tabela.loc[mask, ('Without Role-based', 'Accuracy')] = row['Accuracy']
    df_tabela.loc[mask, ('Without Role-based', 'F1')] = row['F1-score']
    df_tabela.loc[mask, ('', 'Model Used')] = row['Modelo']

    # Preencher With Role-based se houver correspondente
    chave = (strategy_mapped, tipo)
    if chave in piores_role:
        role_row = piores_role[chave]
        df_tabela.loc[mask, ('With Role-based', 'Pr')] = role_row['Precision']
        df_tabela.loc[mask, ('With Role-based', 'Re')] = role_row['Recall']
        df_tabela.loc[mask, ('With Role-based', 'Accuracy')] = role_row['Accuracy']
        df_tabela.loc[mask, ('With Role-based', 'F1')] = role_row['F1-score']

# Calcular diferenças
df_tabela[('', 'Diff_Pr')] = df_tabela[('With Role-based', 'Pr')] - df_tabela[('Without Role-based', 'Pr')]
df_tabela[('', 'Diff_Re')] = df_tabela[('With Role-based', 'Re')] - df_tabela[('Without Role-based', 'Re')]
df_tabela[('', 'Diff_Accuracy')] = df_tabela[('With Role-based', 'Accuracy')] - df_tabela[('Without Role-based', 'Accuracy')]
df_tabela[('', 'Diff_F1')] = df_tabela[('With Role-based', 'F1')] - df_tabela[('Without Role-based', 'F1')]

# Visualização rápida
print(df_tabela.head())

# Salvar em Excel com cabeçalhos
df_tabela.to_excel("granularidade_fina/results_table/rq2_piores.xlsx", index=True)
