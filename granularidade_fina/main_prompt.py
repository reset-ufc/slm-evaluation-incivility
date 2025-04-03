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
from concurrent.futures import ProcessPoolExecutor, TimeoutError

#dotenv.load_dotenv()

#key = os.getenv('OPENAI_KEY')

#client_openAI = openai.OpenAI(api_key=key)

def classify_pr_with_openai(user_prompt, model,prompt_strategy):
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
        response_format={
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
        
    )

    response = json.loads(completion.choices[0].message.content)
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
def classify_pr_with_timeout(user_prompt, model, prompt_strategy,timeout=60):
    """Executa `classify_pr` com um limite de tempo usando processos."""
    try:
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(classify_pr, user_prompt, model,prompt_strategy)
            try:
                response = future.result(timeout=timeout)
                response = response['response']
                type(response)
                print(response)
                response = json.loads(response)
                print(f'Esse foi o response: {response}')
                print(f'Tipo da resposta: {type(response)}')
                return response
            except TimeoutError:
                print("Tempo limite excedido. Terminando processo...")
                for pid, process in executor._processes.items():
                    os.kill(pid, signal.SIGTERM)
                return {"label": "timeout"}
    except Exception as e:
        print(f"Unexpected error in classify_pr_with_timeout: {e}")
        return {"label": "error"}


# Classe do classificador
class Classifier:
    def __init__(self, model, texts):
        self.model = model
        self.texts = texts

    def run_classifier(self, system_strategy, tbdf_new):
        for i, valor in enumerate(tqdm(self.texts, desc=f"Processing {self.model}", unit="text")):
            prompt = prompt_factory(valor,system_strategy)
            prompt = prompt['user_msg']
            response = classify_pr_with_timeout(prompt, self.model, system_strategy,timeout=60)
            if "label" in response:
                tqdm.write(f'example {i}: {response["label"]}')
                tbdf_new.append(response['label'])
            else:
                tqdm.write(f'example {i}: Key "label" not found in response: {response}')
                tbdf_new.append('Model failed to classify')
        return tbdf_new

#runner
def classificador_runner(model_list, strategies, comments):
    total_tasks = len(model_list) * len(strategies)
    with tqdm(total=total_tasks, desc="Overall Progress", unit="task") as progress_bar:
        for system_strategy in strategies:
            for modelo in model_list:
                try:
                    sanitized_model_name = re.sub(r'[^\w\-_]', '_', modelo)
                    classifier_instance = Classifier(modelo, comments)
                    tbdf_new = []
                    tbdf_new = classifier_instance.run_classifier(system_strategy, tbdf_new)
            
                    # Salvar resultados no CSV
                    num_items = len(tbdf_new)
                    aligned_comments = comments[:num_items]
                    df = pd.DataFrame({'Comments': aligned_comments, 'Tbdf': tbdf_new})
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
                    # Cria a pasta (e subpastas) se não existirem ainda
                    os.makedirs(folder_path, exist_ok=True)
                    csv_filename = os.path.join(folder_path,f"fineclassify_{sanitized_model_name}_{system_strategy}.csv")
                    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')

                    progress_bar.update(1)
                except Exception as e:
                    print(f"Error processing model {modelo} with prompt {system_strategy}: {e}")
                    continue

# Proteção para subprocessos no Windows
if __name__ == "__main__":
    # Carregar dataset
    reference_dataset_fg = pd.read_csv("data/reference_dataset_fg.csv")
    comments = reference_dataset_fg['comment_body'].dropna()

    
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
    'llama3.1:8b'
]

    # Executar classificadores
    
    classificador_runner(model_list, strategies, comments)
    print("Processamento concluído.")
