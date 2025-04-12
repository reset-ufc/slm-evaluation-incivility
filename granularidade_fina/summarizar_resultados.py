import os
import pandas as pd
import re
from sklearn.metrics import classification_report
from pathlib import Path

class GeradorResultados:
    def __init__(self, tbdf_classes):
        self.tbdf_classes = tbdf_classes
        self.tbdf_classes_set = set(tbdf_classes)
        self.tbdf_aliases = {
            "identify attacks/name calling": [
                r"^identify attack$",             # Exato: "identify attack"
                r"^identify attacks$",            # Exato: "identify attacks"
                r"^identify attack \(ia\)$",      # Exato: "identify attack (ia)"
                r"^identify attacks \(ia\)$",     # Exato: "identify attacks (ia)"
                r"^identify attack\/ia$",         # Exato: "identify attack/ia" (escapando o /)
                r"^identify attacks\/ia$",        # Exato: "identify attacks/ia" (escapando o /)
                r"^identify attack/name calling$", # Exato: "identify attack/name calling" (escapando o /)
                "ia (name calling)",              # Alias literal
                r"^ia - identify attack$",        # Exato: "ia - identify attack"
                r"^ia$",                          # Exato: "ia"
            ],

        }

    def ler_resultados_csv(self, pasta=".", prefixo="fineclassify_"):
        resultados = {}
        print(f"essa é a pasta: ", pasta)

        for path in Path('./results').rglob('*.csv'):
            file_path = path.resolve()

            nome_strategy = path.parent.parent.name
            nome_modelo = path.parent.name
            nome_modelo_strategy = f"{nome_modelo}_{nome_strategy}"

            if not path.name.startswith(prefixo):
                continue

            print(f"Lendo: {file_path} (modelo: {nome_modelo} | strategy: {nome_strategy})")
            df = pd.read_csv(file_path)

            resultados[nome_modelo_strategy] = df  # Apenas 1 DataFrame por chave

        return resultados

    def extrair_classe(self, texto):
        texto = texto.lower().strip()

        # Verifica se texto bate com algum alias conhecido
        for classe, aliases in self.tbdf_aliases.items():
            if texto == classe:
                return classe
            for alias in aliases:
                if re.fullmatch(alias, texto):  # Usando fullmatch para correspondência exata
                    return classe

        # Verificação direta com as classes conhecidas
        for tbdf in self.tbdf_classes:
            if texto == tbdf:
                return tbdf
            # Verifica se o texto contém a classe com asteriscos, se necessário
            padrao_asteriscos = re.escape(f"**{tbdf}**")
            if re.search(padrao_asteriscos, texto):  # Continua utilizando search para asteriscos
                return tbdf

        return "desconhecido"



    def analisar_resultados(self, pasta=".", salvar_csv=False):
        
        resultados = self.ler_resultados_csv(pasta)
        print(resultados)
        lista_resultados = [] 
        pasta_saida = "results"
        
        for nome_arquivo_csv, df_pred in resultados.items():
            print(f"\nAnalisando: {nome_arquivo_csv}")
            reference_dataset_fg = pd.read_csv("./data/reference_dataset_fg.csv", na_values=[""], keep_default_na=False, dtype={'tbdf': str})
            
            reference_dataset_fg["tbdf"] = reference_dataset_fg["tbdf"].astype(str).str.strip().str.lower()
            df_pred["Tbdf"] = df_pred["Tbdf"].fillna("None").astype(str).str.strip().str.lower()
            reference_dataset_fg = reference_dataset_fg.drop_duplicates(subset="comment_body")
            df_pred = df_pred.drop_duplicates(subset="Comments")

            
            df_merged = reference_dataset_fg.merge(df_pred, left_on="comment_body", right_on="Comments", how="inner")
            print(df_merged.columns)
            print("Qtd de linhas em reference_dataset_fg:", len(reference_dataset_fg))
            print("Qtd de linhas em df_pred:", len(df_pred))
            print("Qtd de linhas em df_merged:", len(df_merged))
            y_true = df_merged['tbdf'].astype(str)
            y_pred = df_merged['Tbdf'].astype(str)

            print("Rótulos únicos em y_true:", set(y_true.unique()))
            print("Rótulos únicos em y_pred:", set(y_pred.unique()))          
                
            # Generate class report in dic format
            report_dict = classification_report(y_true, y_pred, labels=self.tbdf_classes, zero_division=0, output_dict=True)

            # Create dic with results per class
            custom_report = {}

            for label in self.tbdf_classes:
                real_samples_forlabel = (y_true == label).sum()
                predicted_samples_forlabel = (y_pred == label).sum()
                predicted_errors = (y_pred == 'desconhecido').sum()
                predicted_samples = len(y_pred)
                false_positives = ((y_pred == label) & (y_true != label)).sum()
                false_negatives = ((y_true == label) & (y_pred != label)).sum()
                true_positives = ((y_true == label) & (y_pred == label)).sum()
                
                # Cálculo manual da acurácia por classe
                if real_samples_forlabel > 0:
                    accuracy = true_positives / real_samples_forlabel
                else:
                    accuracy = 0.0

                custom_report[label] = {
                    'Precision': report_dict[label]['precision'],
                    'Recall': report_dict[label]['recall'],
                    'Accuracy': accuracy,
                    'F1-score': report_dict[label]['f1-score'],
                    'Real_samples': real_samples_forlabel,
                    'Predicted_samples': predicted_samples_forlabel,
                    'Fp': false_positives,
                    'Fn': false_negatives, 
                    'Allucination Rate': (predicted_errors / predicted_samples)*100
                }
            partes = nome_arquivo_csv.split('_')
            modelo = '_'.join(partes[:2])
            estrategia = '_'.join(partes[2:])
            
            # Normaliza o nome do arquivo para usar underscores
            nome_base = nome_arquivo_csv.replace('.csv', '').replace('-', '_')

            # Dicionário corrigido com os nomes desejados para as pastas
            modelos_pasta = {
                "gemma_7b": "gemma_7b",
                "gemma2_9b": "gemma2_9b",
                "mistral_7b": "mistral_7b",
                "deepseek_8b": "deepseek-8b",  # Nome ajustado para a pasta
                "deepseek_14b": "deepseek-14b",  # Nome ajustado para a pasta
                "llama3.2_3b": "llama3.2_3b",
                "mistral_nemo_12b": "mistral-nemo_12b",
                "llama3.1_8b": "llama3.1_8b",
                "phi3_3.8b": "phi3_3.8b",
                "phi4_14b": "phi4_14b"
            }

            # Encontra qual modelo está no nome do arquivo
            modelo_escolhido = next((modelo for modelo in modelos_pasta.keys() if modelo in nome_base), "outros")
                
            if "deepseek_8b" in modelo_escolhido:
                modelo = "deepseek-8b"
            if "deepseek_14b" in modelo_escolhido:
                modelo = "deepseek-14b"
            # Se um modelo foi encontrado, removemos ele do nome_base para obter a estratégia corretamente
            if modelo_escolhido != "outros":
                estrategia = nome_base.replace(modelo_escolhido, '').strip('_')
            else:
                estrategia = nome_base  # Se nenhum modelo for identificado, toda a string será tratada como estratégia

            # Define o caminho de saída corretamente
            pasta_saida = f"results/{estrategia}/{modelos_pasta.get(modelo_escolhido, 'outros')}"

            print(pasta_saida)  # Apenas para testar o resultado


            
                
            # Convert dic in df and add model name column
            os.makedirs(pasta_saida, exist_ok=True)
            df_result = pd.DataFrame(custom_report).transpose()
            df_result.reset_index(inplace=True)
            df_result.rename(columns={'index':'Tbdf'}, inplace=True)
            df_result.insert(0, "Modelo", modelo)
            df_result.insert(0, "Strategy", estrategia)
            df_result.to_csv(f"{pasta_saida}/{nome_arquivo_csv}_result.csv", index=False)


            lista_resultados.append(df_result)

        if not resultados:
            print("Nenhum resultado encontrado!")
            

        print(lista_resultados)
        df_final = pd.concat(lista_resultados,ignore_index=True)
        nome_arquivo = "results_concat.csv"
        pasta_concat = "./results_table"
        df_final.to_csv(os.path.join(pasta_concat, nome_arquivo), index=True)  
        print("Duplicatas no df_pred (baseado na coluna 'Comments'):", df_pred.duplicated(subset='Comments').sum())
        print("Duplicatas no reference_dataset_fg (baseado em 'comment_body'):", reference_dataset_fg.duplicated(subset='comment_body').sum())
        print("Duplicatas no df_merged (baseado em 'comment_body'):", df_merged.duplicated(subset='comment_body').sum())
        



        print(f"\nResultados salvos em: {os.path.join(pasta_saida, nome_arquivo)}")
        
        return df_final


# Definição das classes tbdf
tbdf_classes = ['bitter frustration', 'impatience', 'mocking', 'irony', 'vulgarity', 'threat', 'identify attack/name calling', 'none', 'entitlement','insulting']

# INIT SCRIPT FOR READ CSV FILES
root_dir = 'results'
csv_dfs = []

# Read pasts and sub pasts
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(root, file)
            print(f"Lendo: {file_path}")
            df = pd.read_csv(file_path)
            csv_dfs.append(df)

# SCRIPT FOR READ CSV FILES FINALIZED


# Generate results in different pasts
gerador = GeradorResultados(tbdf_classes)
gerador.analisar_resultados(pasta="./results", salvar_csv=True)
