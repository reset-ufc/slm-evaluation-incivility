import pandas as pd

def read_csv_file():
    df = pd.read_csv(r'granularidade_fina\results_table\results_concat.csv')
    
    labels = {
        'bitter Frustration': 'Bitter Frustration',
        'impatience': 'Impatience',
        'mocking': 'Mocking',
        'vulgarity': 'Vulgarity',
        'insulting': 'Insulting',
        'none': 'None',
        'entitlement': 'Entitlement',
        'ia (name calling)': 'Identity Attacks',
        'irony': 'Irony',
        'threat': 'Threat'
    }
    
    strategies = ['zero_shot', 'one_shot', 'few_shot_3', 'role_based', 'auto_cot', 'role_based_one_shot', 'role_based_few_shot_3', 'role_based_auto_cot']
    
    # Criando uma estrutura para armazenar os dados
    data = []
    
    for strategy in strategies:
        if 'auto_cot' not in strategy:
            for key, label in labels.items():
                df_filtered = df[(df['Tbdf'] == key) & (df['Strategy'] == strategy)]
                
                if not df_filtered.empty:
                    row = [strategy, label] + df_filtered[['Precision', 'Recall', 'F1-score', 'Fp', 'Fn']].values.flatten().tolist() * 2
                else:
                    row = [strategy, label] + ["" for _ in range(10)]
                
                data.append(row)
        else:
            
    
    # Criando o DataFrame
    columns = [
        "Strategy", "Type",
        "Pr (Without Role-based)", "Re (Without Role-based)", "F1 (Without Role-based)", "FP (Without Role-based)", "FN (Without Role-based)",
        "Pr (With Role-based)", "Re (With Role-based)", "F1 (With Role-based)", "FP (With Role-based)", "FN (With Role-based)"
    ]
    
    df_final = pd.DataFrame(data, columns=columns)
    
    # Salvando como CSV
    df_final.to_csv("tabela_resultados.csv", index=False)
    
    return df_final

# Executa a leitura e preenchimento
df_resultado = read_csv_file()
print(df_resultado)