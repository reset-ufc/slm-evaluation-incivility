import pandas as pd
import ollama
from ollama import chat, Options
from prompts import prompt_factory
import json
from tqdm import tqdm  # Barra de progresso
import re
import openai
import dotenv
import time
import os
import signal
import tiktoken
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import csv
from pathlib import Path

dotenv.load_dotenv()

key = os.getenv('OPENAI_KEY2')

client_openAI = openai.OpenAI(api_key=key)

def classify_pr_with_openai(user_prompt, model,prompt_strategy):
    if "auto_cot" not in prompt_strategy:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "tone_classification",
                "schema": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "enum": ["Bitter Frustration","Vulgarity","Impatience","Mocking","Irony","Entitlement","Insulting","Threat","Identify Attack/Name Calling","None"]
                        }
                    },
                    "required": ["label"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    
    else:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "tone_classification",
                "schema": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "enum": ["Bitter Frustration","Vulgarity","Impatience","Mocking","Irony","Entitlement","Insulting","Threat","Identify Attack/Name Calling","None"]
                        },
                        "reasoning": {
                            "type": "string"
                        }
                    },
                    "required": ["label", "reasoning"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }

    """Generates classification from the model."""
    completion = client_openAI.chat.completions.create(
        model=model,
        messages=[
        {
            "role": "user",
            "content": f"{user_prompt}"
        }
        ],
        n=1,
        temperature=0.0,
        response_format=response_format
        
    )

    comp = completion.choices[0].message.content
    print(comp)
    try: 
        response = json.loads(comp)
    except json.JSONDecodeError as e:
        print(f"Erro: {e}")
        print(comp[e.pos-50:e.pos+50])
    return response

def classify_pr(user_prompt, model,prompt_strategy):
    
    if "auto_cot" not in prompt_strategy:
        response_format = {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "enum": ["Bitter Frustration","Vulgarity","Impatience","Mocking","Irony","Entitlement","Insulting","Threat","Identify Attack/Name Calling","None"]
                }
            },
            "required": [
            "label"
            ]
        }
    
    else:
        response_format = {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "enum": ["Bitter Frustration","Vulgarity","Impatience","Mocking","Irony","Entitlement","Insulting","Threat","Identify Attack/Name Calling","None"]
                },
                "reasoning": {
                    "type": "string"
                }
            },
            "required": [
            "label"
            ]
            }
    if 'gpt' in model:
        return classify_pr_with_openai(user_prompt, model,prompt_strategy)
    try:
        return ollama.generate(
        model=model,
        format=response_format,
        options={
            "temperature": 0,
        },
        stream=False,
        prompt=f"""{user_prompt}"""
    )
    except Exception as e:
        print(f"Erro ao rodar o modelo {model}: {e}")
        return {"label": "error"}


# Função com timeout
def classify_pr_with_timeout(user_prompt, model, prompt_strategy,timeout=60*20):
    """Executa `classify_pr` com um limite de tempo usando processos."""
    try:
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(classify_pr, user_prompt, model,prompt_strategy)
            try:
                response = future.result(timeout=timeout)
                if model == "gpt-4o-mini":
                    print(f'Esse foi o response: {response}')
                    #print(f'Tipo da resposta: {type(response)}')
                    return response
                response = response['response']
                # type(response)
                # print(response)
                try:
                    response = json.loads(response)
                except json.JSONDecodeError as e:
                    print(f"Erro: {e}")
                    print(response[e.pos-50:e.pos+50])
                #print(f'Esse foi o response: {response}')
                #print(f'Tipo da resposta: {type(response)}')
                return response
            except TimeoutError:
                print("Tempo limite excedido. Terminando processo...")
                for pid, process in executor._processes.items():
                    os.kill(pid, signal.SIGTERM)
                return {"label": "timeout"}
    except Exception as e:
        print(f"Unexpected error in classify_pr_with_timeout: {e}")
        return {"label": "error"}

encoder = tiktoken.encoding_for_model("gpt-4o-mini")
MAX_TOKENS_PER_MINUTE = 200_000
RESET_INTERVAL = 60

def count_tokens(prompt,):
    """
    Calcula a quantidade de tokens usados no prompt e na resposta.
    """
    prompt_tokens = len(encoder.encode(prompt))

    return prompt_tokens 

def rate_limited_call(prompt, model, strategy, current_tokens, last_reset_time):
    """
    Calls the rate-controlled sort function using the official token count.
    Resets the token counter every minute regardless of whether the limit is reached.
    """
    # Verificar se já passou 1 minuto desde o último reset
    current_time = time.time()
    if current_time - last_reset_time >= RESET_INTERVAL:
        print("Renewing rate limit window (1 minute passed)...")
        current_tokens = 0
        last_reset_time = current_time
    
    tokens_used = count_tokens(prompt)
    current_tokens += tokens_used
    
    # Verificar se vai exceder o limite
    if current_tokens > MAX_TOKENS_PER_MINUTE:
        # Calcular tempo necessário para esperar até completar o minuto
        wait_time = RESET_INTERVAL - (current_time - last_reset_time)
        if wait_time > 0:
            print(f"Token limit reached. Waiting {wait_time:.2f} seconds for renewal...")
            time.sleep(wait_time)
            current_tokens = tokens_used  # Após esperar, contabilizar apenas os tokens desta chamada
            last_reset_time = time.time()  # Atualizar o timestamp de reset
        else:
            # Se já passou o tempo do reset, apenas atualizar os valores
            current_tokens = tokens_used
            last_reset_time = current_time
    
    response = classify_pr_with_timeout(prompt, model, strategy)
    
    if response is None:
        return None, current_tokens, last_reset_time
    
    return response, current_tokens, last_reset_time

def carregar_lista_json(arquivo):
    import json, os
    if os.path.exists(arquivo):
        try:
            with open(arquivo, "r", encoding="utf-8") as f:
                conteudo = f.read().strip()
                if not conteudo:
                    return []
                dados = json.loads(conteudo)
                if isinstance(dados, list):
                    return dados
        except json.JSONDecodeError:
            print(f"[Aviso] O arquivo {arquivo} está corrompido ou mal formatado.")
            return []
    return []

def adicionar_dado_em_lista_json(arquivo, novo_dado):
    dados = carregar_lista_json(arquivo)
    dados.append(novo_dado)
    with open(arquivo, "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=4, ensure_ascii=False)

# Classe do classificador
class Classifier:
    def __init__(self, model, texts):
        self.model = model
        self.texts = texts

    def run_classifier(self, system_strategy, tbdf_new, path_csv, reasoning_path):
        current_tokens = 0 
        last_reset_time = time.time()  # Iniciar o timestamp

        with tqdm(total=len(self.texts), desc=f"Progress for {self.model} in {system_strategy}", unit="task") as progress_bar:
            for i, row in self.texts.iterrows():
                prompt = prompt_factory(row['comment_body'],system_strategy)
                prompt = prompt['user_msg']
                if 'gpt' in self.model:
                    response, current_tokens, last_reset_time = rate_limited_call(prompt, self.model, system_strategy, current_tokens, last_reset_time)
                else:
                    try: 
                        response = classify_pr_with_timeout(prompt, self.model, system_strategy, timeout=60)
                        print(response)
                    except Exception as e:
                        print(f"Erro {e} no index {i}.")
                        continue

                if "label" in response:
                    try:
                        tbdf_new.append(response['label'])
                    except Exception as e:
                        print(row['comment_body'])
                        print(f"Erro {e} no index {i}.")
                        continue

                    if response['label'] == "error":
                        continue

                    #tqdm.write(f'example {i}: {response["label"]}')
                    

                    with open(path_csv, mode="a", newline="", encoding="utf-8") as arquivo_csv:
                        escritor = csv.writer(arquivo_csv)
                        escritor.writerow([i, row['comment_body'], response['label']])
                    
                    progress_bar.update(1)
                    
                    # if response['label'] == "timeout":
                    #    continue

                    if 'auto_cot' in system_strategy:
                        adicionar_dado_em_lista_json(reasoning_path, {
                            "index": i,
                            "message": row['comment_body'],
                            "strategy": system_strategy,
                            "model": self.model,
                            "reasoning": response['reasoning']
                        })
                    

                    
                else:
                    tqdm.write(f'example {i}: Key "label" not found in response: {response}')
                    tbdf_new.append('Model failed to classify')

                    with open(path_csv, mode="a", newline="", encoding="utf-8") as arquivo_csv:
                        escritor = csv.writer(arquivo_csv)
                        escritor.writerow([i, row['comment_body'], 'Model failed to classify'])
                    print(response)
                    #break
            return tbdf_new

#runner
def classificador_runner(model_list, strategies, comments):
    total_tasks = len(model_list) * len(strategies)
    with tqdm(total=total_tasks, desc="Overall Progress", unit="task") as progress_bar:
        for system_strategy in strategies:
            for modelo in model_list:
                try:
                    sanitized_model_name = re.sub(r'[^\w\-_]', '_', modelo)
            
                    # Salvar resultados no CSV
                    # num_items = len(tbdf_new)
                    # aligned_comments = comments[:num_items]
                    # df = pd.DataFrame({'Comments': aligned_comments, 'Tbdf': tbdf_new})
                    # Caminho da pasta de destino
                    if modelo == "gemma:7b":
                        folder_path = f"results/{system_strategy}/gemma_7b"
                    elif modelo == "gemma2:9b":
                        folder_path = f"results/{system_strategy}/gemma2_9b"
                    elif modelo == "mistral-nemo:12b":
                        folder_path = f"results/{system_strategy}/mistral-nemo_12b"
                    elif modelo == "mistral:7b":
                        folder_path = f"results/{system_strategy}/mistral_7b"
                    elif modelo == "deepseek-r1:8b":
                        folder_path = f"results/{system_strategy}/deepseek-8b"
                    elif modelo == "deepseek-r1:14b":
                        folder_path = f"results/{system_strategy}/deepseek-14b"
                    elif modelo == "llama3.2:3b":
                        folder_path = f"results/{system_strategy}/llama3.2_3b"
                    elif modelo == "llama3.1:8b":
                        folder_path = f"results/{system_strategy}/llama3.1_8b"
                    elif modelo == "gpt-4o-mini":
                        folder_path = f"results/{system_strategy}/gpt-4o-mini"
                    
                    csv_filename = os.path.join(folder_path,f"fineclassify_{sanitized_model_name}_{system_strategy}.csv")

                    if Path(csv_filename).exists():
                        
                        all_idxs = set(list(range(len(comments))))
                        preds = pd.read_csv(csv_filename)
                        preds = preds.loc[~preds['Tbdf'].isin(['timeout', 'error'])]


                        idxs_classified = set(preds['index'].to_list())
                        index_to_classify = list(all_idxs - idxs_classified)
                        comments_to_classify = comments.iloc[index_to_classify]
                        #print(index_to_classify)

                    else:
                        comments_to_classify = comments.copy()


                    classifier_instance = Classifier(modelo, comments_to_classify)
                    tbdf_new = []
                    # Cria a pasta (e subpastas) se não existirem ainda

                    os.makedirs(folder_path, exist_ok=True)

                    reasoning_path = os.path.join(folder_path,f"fineclassify_{sanitized_model_name}_{system_strategy}_reasonings.json")

                    try:
                        with open(csv_filename, mode="x", newline="", encoding="utf-8") as arquivo_csv:
                            escritor = csv.writer(arquivo_csv)
                            escritor.writerow(["index", "Comments", "Tbdf"])
                    except FileExistsError:
                        pass
                    
                    tbdf_new = classifier_instance.run_classifier(system_strategy, tbdf_new, csv_filename, reasoning_path)
                    # df.to_csv(csv_filename, index=False, encoding='utf-8-sig')

                    progress_bar.update(1)
                except Exception as e:
                    print(f"Error processing model {modelo} with prompt {system_strategy}: {e}")
                    #continue
            #if tbdf_new[-1] == "error":
            #    break

# Proteção para subprocessos no Windows
if __name__ == "__main__":
    # Carregar dataset
    reference_dataset_fg = pd.read_csv("data/reference_dataset_fg.csv")
    comments = reference_dataset_fg.dropna(subset=['comment_body'])

    # comments = comments[:1]
    
    # Configurações
    strategies = ['zero_shot','one_shot','few_shot_3', 'role_based', 'auto_cot','role_based_one_shot','role_based_few_shot_3','role_based_auto_cot']
    model_list = [
     'gemma:7b',
     'gemma2:9b', 
     'mistral-nemo:12b', 
     'mistral:7b', 
     'deepseek-r1:8b', 
     'deepseek-r1:14b', 
     'llama3.2:3b', 
     'llama3.1:8b',
     'phi4:14b',
     "gpt-4o-mini"
]

    # Executar classificadores
    
    classificador_runner(model_list, strategies, comments)
    print("Processamento concluído.")
