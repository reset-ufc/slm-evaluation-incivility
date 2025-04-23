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
            "required": ["label", "reasoning"]
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

def classify_with_timeout(user_prompt, model, prompt_strategy, timeout=60*30):
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
    
    response = classify_with_timeout(prompt, model, strategy)
    
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

def main(): 
    data = pd.read_csv(data_dir)
    # data = data.iloc[5:, :]

    n_samples = len(data)
    # todos os idices
    all_indexs = set(range(len(data)))
    # n_samples = 5

    # models = ['gemma:7b', 'gemma2:9b', 'mistral-nemo:12b', 'mistral:7b', 'deepseek-r1:8b', 'deepseek-r1:14b', 'llama3.2:3b', 'llama3.1:8b', 'gpt-4o-mini', 'phi4:14b']

    models = ['phi4:14b']
    current_tokens = 0 
    last_reset_time = time.time()  # Iniciar o timestamp
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
            
            predictions_path = strategy_path / 'predictions_df.csv'

            if predictions_path.exists():
                pred_df = pd.read_csv(predictions_path, encoding="utf-8")
            
                # idices classificados
                indexes_classified = set(int(i) for i in pred_df['index'].values)
                #print(indexes_classified)

                # idices que ainda não foram classificados
                index_to_classify = list(all_indexs - indexes_classified)
            
                data_to_classify = data.iloc[index_to_classify]
                n_samples = len(data_to_classify)
                # pred_df = pd.DataFrame(columns=cols_pred_df)
                truth, predictions = [], []
                # timeout_exceeded, errors = [], []
            else:
                data_to_classify = data.copy()
                n_samples = len(data)


            with tqdm(total=n_samples, desc=f"Processing strategy '{strategy}' for model '{model}'") as pbar:
                for index, row in data_to_classify.iterrows():
                    if index == n_samples:
                        break
                    user_prompt = prompt_factory(row['message'], strategy)

                    if 'gpt' in model:
                        response, current_tokens, last_reset_time = rate_limited_call(
                            user_prompt['user_msg'], model, strategy, current_tokens, last_reset_time
                        )
                    else:
                        response = classify_with_timeout(user_prompt['user_msg'], model, strategy)

                    if response is None:
                        adicionar_dado_em_lista_json(timeout_file, {
                            "index": index,
                            "message": row['message'],
                            "strategy": strategy,
                            "model": model
                        })
                        pbar.update(1)
                        continue

                    try:
                        if 'gpt' not in model:
                            response = json.loads(response['response'])

                        if 'auto_cot' in strategy:
                            ### VERIFICAR O FORMATO DE RESPOSTA DA API OPENAI QUANDO COT É UTILIZADO
                            ## PEGAR O ARQUIVO DE REASONINGS E """""ADICIONAR""""" A RESPOSTA, pois atualmente esta reescrevendo a resposta
                            adicionar_dado_em_lista_json(reasonings_file, {
                                "index": index,
                                "message": row['message'],
                                "strategy": strategy,
                                "model": model,
                                "reasoning": response['reasoning']
                            })
                        pred = extract_label(response)

                        truth.append(row['actual'])
                        predictions.append(pred)

                        # adicionar as predições no arquivo csv
                        with open(pred_df_file, mode="a", newline="", encoding="utf-8") as arquivo_csv:
                            escritor = csv.writer(arquivo_csv)
                            escritor.writerow([index, row['message'], pred, row['actual'], row['source']])

                    except json.JSONDecodeError as e:
                        print("Execution stopped at index:", index)
                        error = e.model_dump() if hasattr(e, "model_dump") else str(e)
                        adicionar_dado_em_lista_json(errors_file, {
                                            "index": index,
                                            "message": row['message'],
                                            "strategy": strategy,
                                            "model": model,
                                            "error": error
                                        })
                    

                    except Exception as e:
                        print("Execution stopped at index:", index)
                        error = e.model_dump() if hasattr(e, "model_dump") else str(e)
                        adicionar_dado_em_lista_json(errors_file, {
                                            "index": index,
                                            "message": row['message'],
                                            "strategy": strategy,
                                            "model": model,
                                            "error": error
                                        })
                    
                    pbar.update(1)

            print('model:', model)
            print("num valid samples:", len(predictions))

if __name__ == "__main__":
    main()
