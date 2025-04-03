from prompt import prompt_factory, strategies
import json
import ollama
import pandas as pd
from pathlib import Path
import os
import multiprocessing
from tqdm import tqdm
import openai
import dotenv
import csv
import tiktoken
import time

dotenv.load_dotenv()

key = os.getenv('OPENAI_KEY')

client_openAI = openai.OpenAI(api_key=key)

cur_dir = Path(os.getcwd())
data_dir = cur_dir / 'data' / 'final_df.csv'
results_dir = cur_dir / 'results'
results_dir.mkdir(parents=True, exist_ok=True)

def normalize_response(response):
    mapping = {
        "civil": "CIVIL",
        "uncivil": "UNCIVIL",

        # respostas que geram erro
        "civel": "CIVIL",
        "civic": "CIVIL",
        "cive": "CIVIL",
        "civ": "CIVIL",
        "civie": "CIVIL",
        "civul": "CIVIL",
        "civl": "CIVIL",
        "uncivic": "UNCIVIL",
        "uncivel": "UNCIVIL",
        "unciv": "UNCIVIL",
        "uncivie": "UNCIVIL",
        "uncival": "UNCIVIL",
        "cuncivil": "UNCIVIL",
        "cuncil": "UNCIVIL",
    }
    return mapping.get(response.lower(), response)

def classify_pr_with_openai(user_prompt, model, prompt_strategy):
    if 'auto_cot' not in prompt_strategy:
        response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "tone_classification",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "enum": ["CIVIL", "UNCIVIL"]
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
                                "enum": ["CIVIL", "UNCIVIL"]
                            },
                            "reasoning": {
                                "type": "string"
                            }
                        },
                        "required": ["label"],
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

    response = json.loads(completion.choices[0].message.content)
    return response


def classify_pr(user_prompt, model, prompt_strategy):
    """Generates classification from the model."""

    if "auto_cot" not in prompt_strategy:
        response_format = {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "enum": ["CIVIL", "UNCIVIL"]
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
                            "enum": ["CIVIL", "UNCIVIL"]
                        },
                        "reasoning": {
                            "type": "string"
                        }
                    },
                    "required": [
                    "label"
                    ]
                }

    # gpt model
    if 'gpt' in model:
        return classify_pr_with_openai(user_prompt, model, prompt_strategy)
    
    return ollama.generate(
        model=model,
        format=response_format,
        options={
                "temperature": 0,
            },
        stream=False,
        prompt=f"""{user_prompt}"""
    )

def extract_label(response):
    """Extracts the label from the response."""
    mapping_int = {'CIVIL': 0, 'UNCIVIL': 1}

    if 'label' in response:
        class_name = normalize_response(response['label'])
    
    #for few_shot strategy
    if 'response_7' in response:
        response = response['response_7']['label']
        class_name = normalize_response(response)
    
    # for one_shot strategy
    if 'response_3' in response:
        response = response['response_3']['label']
        class_name = normalize_response(response)

    pred = int(mapping_int[class_name])
    return pred

def target_func(user_prompt, model, queue, prompt_strategy):
    try:
        result = classify_pr(user_prompt, model, prompt_strategy)
        queue.put(result)
    except Exception as e:
        queue.put({'error': str(e)})

def classify_with_timeout(user_prompt, model, prompt_strategy, timeout=60):
    """Runs the classify_pr function with a timeout using multiprocessing."""
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=target_func, args=(user_prompt, model, queue, prompt_strategy))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return None

    try:
        return queue.get(timeout=1)  # Timeout evita bloqueio
    except:
        return None


encoder = tiktoken.encoding_for_model("gpt-4")
MAX_TOKENS_PER_MINUTE = 200_000
RESET_INTERVAL = 60

def count_tokens(prompt, response):
    """
    Calcula a quantidade de tokens usados no prompt e na resposta.
    """
    prompt_tokens = len(encoder.encode(prompt))
    
    # Lidar com diferentes formatos de resposta
    if isinstance(response, dict):
        if 'response' in response:
            response_text = response['response']
        else:
            response_text = json.dumps(response)
    else:
        response_text = str(response)
        
    response_tokens = len(encoder.encode(response_text))
    return prompt_tokens + response_tokens

def rate_limited_call(prompt, model, strategy, current_tokens):
    """
    Chama a função de classificação com controle de taxa usando contagem de tokens oficial.
    """
    response = classify_with_timeout(prompt, model, strategy)

    if response is None:
        return None, current_tokens

    tokens_used = count_tokens(prompt, response)
    current_tokens += tokens_used

    # Se exceder o limite, esperar até a janela de tempo ser renovada
    if current_tokens > MAX_TOKENS_PER_MINUTE:
        print("Limite de tokens atingido. Aguardando renovação...")
        time.sleep(RESET_INTERVAL)
        current_tokens = tokens_used  # Resetar contagem após espera

    return response, current_tokens

def main(): 
    data = pd.read_csv(data_dir)

    # n_samples = len(data)
    n_samples = 1

    # models = ['gemma:7b', 'gemma2:9b', 'mistral-nemo:12b', 'mistral:7b', 'deepseek-r1:8b', 'deepseek-r1:14b', 'llama3.2:3b', 'llama3.1:8b']

    models = ['gpt-4o-mini']
    current_tokens = 0 
    #models = ['gemma3:1b']

    for model in models:
        model_path = results_dir / model.replace(':', '_')
    
        model_path.mkdir(parents=True, exist_ok=True)

        for strategy in strategies:
            strategy_path = model_path / strategy
            strategy_path.mkdir(parents=True, exist_ok=True)

            # criar o arquivo de predicoes em csv
            pred_df_file = strategy_path / 'predictions_df.csv'
            errors_file = strategy_path / 'errors.json'
            timeout_file = strategy_path / 'timeout_exceeded.json'
            if 'auto_cot' in strategy:
                # reasonings = []
                reasonings_file = strategy_path / 'reasonings.json'

            cols_pred_df = ['index', 'message', 'prediction', 'actual', 'source']
            # pred_df = pd.DataFrame(columns=cols_pred_df)
            try:
                with open(pred_df_file, mode="x", newline="", encoding="utf-8") as arquivo_csv:
                    escritor = csv.writer(arquivo_csv)
                    escritor.writerow(cols_pred_df)
            except FileExistsError:
                pass

            print('model:', model)
            print('strategy:', strategy)
            
            # pred_df = pd.DataFrame(columns=cols_pred_df)
            truth, predictions = [], []
            # timeout_exceeded, errors = [], []

            with tqdm(total=n_samples, desc=f"Processing strategy '{strategy}' for model '{model}'") as pbar:
                for index, row in data.iterrows():
                    if index == n_samples:
                        break
                    user_prompt = prompt_factory(row['message'], strategy)

                    if 'gpt' in model:
                        response, current_tokens = rate_limited_call(user_prompt['user_msg'], model, strategy, current_tokens)
                    else:
                        response = classify_with_timeout(user_prompt['user_msg'], model, strategy)

                    if response is None:
                        with open(timeout_file, 'w') as f:
                            json.dump({"index": index, "message": row['message'], "strategy": strategy, "model": model}, f, indent=4)
                        pbar.update(1)
                        continue

                    try:
                        if 'gpt' not in model:
                            response = json.loads(response['response'])

                        if 'auto_cot' in strategy:
                            ### VERIFICAR O FORMATO DE RESPOSTA DA API OPENAI QUANDO COT É UTILIZADO
                            with open(reasonings_file, mode="w", encoding="utf-8") as arquivo_json:
                                json.dump({"index": index, "message": row['message'], "strategy": strategy, "model": model, "reasoning": response['reasoning']},
                                           arquivo_json,
                                            indent=4)

                        pred = extract_label(response)

                        truth.append(row['actual'])
                        predictions.append(pred)

                        # adicionar as predições no arquivo csv
                        with open(pred_df_file, mode="a", newline="", encoding="utf-8") as arquivo_csv:
                            escritor = csv.writer(arquivo_csv)
                            escritor.writerow([index, row['message'], pred, row['actual'], row['source']])

                    except json.JSONDecodeError as e:
                        print("Execution stopped at index:", index)
                        with open(errors_file, mode="w", encoding="utf-8") as arquivo_json:
                            error = e.model_dump() if hasattr(e, "model_dump") else str(e)
                            json.dump({'index': index, 'message': row['message'], 'strategy': strategy, 'model': model, "error": error, 'response': response}, arquivo_json, indent=4)
                    

                    except Exception as e:
                        print("Execution stopped at index:", index)
                        with open(errors_file, mode="w", encoding="utf-8") as arquivo_json:
                            error = e.model_dump() if hasattr(e, "model_dump") else str(e)
                            json.dump({'index': index, 'message': row['message'], 'strategy': strategy, 'model': model, "error": error, 'response': response}, arquivo_json, indent=4)
                    
                    pbar.update(1)

            print('model:', model)
            print("num valid samples:", len(predictions))

if __name__ == "__main__":
    main()
