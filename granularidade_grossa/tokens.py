import pandas as pd
import tiktoken
from prompt import prompt_factory

strategies = ['zero_shot', 'one_shot', 'few_shot', 'auto_cot', 'role_based', 'role_based_few_shot', 'role_based_auto_cot', 'role_based_one_shot']

# Função para contar tokens
def contar_tokens(texto, modelo="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(modelo)
    return len(encoding.encode(texto))

# Carregando o dataset
data = pd.read_csv(r"C:\Users\mario\Documents\estudos\ufc\LLM-TESTS\granularidade_grossa\data\final_df_cleaned.csv")

for strategy in strategies:
    print(f"strategy: {strategy}")

    # Criando uma nova coluna com o prompt
    data[strategy] = data['message'].apply(lambda x: prompt_factory(x, strategy=strategy))

    # Calculando tokens e criando nova coluna
    data['token_count'] = data[strategy].apply(lambda x: contar_tokens(str(x)))

    print("maximo de tokens: ", data['token_count'].max())
    print("minimo de tokens: ", data['token_count'].min())

    print('media de tokens: ', data['token_count'].mean())
    print('desvio padrao de tokens: ', data['token_count'].std())