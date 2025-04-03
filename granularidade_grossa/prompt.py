import json
from pathlib import Path

file_path = Path(r'data\exemplos.json')

# Ler os exemplos
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Acessar os exemplos
civil_examples = data["civil"]
uncivil_examples = data["uncivil"]

def prompt_factory(input_text, strategy):
    sys_msg = ''
    user_msg = ''
    task = "to determine if the message is **CIVIL** or **UNCIVIL**"
    answer_template_specification = f"""Return the result as a JSON with the following format: {{'label': 'CIVIL' OR 'UNCIVIL'}}"""
    model_instruction = "**You are a GitHub moderator of conversations** that classifies the tone of messages in GitHub discussions as CIVIL or UNCIVIL"
    cot_promotion = """Let's think step by step. Provide reasoning before giving the response"""
    answer_template_specification_auto_cot = """Your response should be only a json in the following format: {{"label": "CIVIL" or "UNCIVIL", "reasoning": "Your reasoning here"}}"""
    civil_example1 = civil_examples[0]
    civil_example2 = civil_examples[1]
    civil_example3 = civil_examples[2]
    uncivil_example1 = uncivil_examples[0]
    uncivil_example2 = uncivil_examples[1]
    uncivil_example3 = uncivil_examples[2]
    task_instruction = "CIVIL messages are those that are respectful, constructive, and polite. UNCIVIL messages include any of the following tones: Mocking which involves ridiculing or making fun of someone in a disrespectful way; Identity Attack or name-calling which consists of making derogatory comments based on race, religion, gender, sexual orientation, or nationality; Bitter Frustration which expresses strong frustration, displeasure, or annoyance; Impatience which conveys dissatisfaction due to delays; Threat which involves issuing a warning that implies a negative consequence; Irony which uses language to imply a meaning opposite to the literal one, often sarcastically; Vulgarity which includes offensive or inappropriate language; Entitlement which reflects an expectation of special treatment or privileges; and Insulting which involves making derogatory remarks towards another person or project"

    if strategy == 'zero_shot':
        user_msg = f"""Consider the following message: "{input_text}". Your task is {task}. {answer_template_specification}."""
    
    if strategy =='one_shot':
        user_msg = f"""Your task is {task}. {answer_template_specification}.
                        Consider the following example messages:
                        Example 1: "{civil_example1}". \
                        Response for Example 1: {{"label":"CIVIL"}}
                        Example 2: {uncivil_example1} \
                        Response for Example 2: {{"label":"UNCIVIL"}}
                        Now, analyze the following message:
                        "{input_text}" """

    if strategy =='few_shot':
        user_msg = f"""Your task is {task}. {answer_template_specification}.
                    Consider the following example messages:
                    Example 1: "{civil_example1}". \
                    Response for Example 1: {{"label":"CIVIL"}}
                    Example 2: {civil_example2} \
                    Response for Example 2: {{"label":"CIVIL"}}
                    Example 3: "{civil_example3}". \
                    Response for Example 3: {{"label":"CIVIL"}}
                    Example 4: {uncivil_example1} \
                    Response for Example 4: {{"label":"UNCIVIL"}}
                    Example 5: "{uncivil_example2}". \
                    Response for Example 5: {{"label":"UNCIVIL"}}
                    Example 6: {uncivil_example3} \
                    Response for Example 6: {{"label":"UNCIVIL"}}
                    Now, analyze the following message:
                    "{input_text}" """

    if strategy == "auto_cot":
        user_msg = f"""Consider the following message: "{input_text}". Your task is {task}. {cot_promotion}. {answer_template_specification_auto_cot}"""
    
    if strategy == 'role_based':
        user_msg = f"""{model_instruction}. {task_instruction}. Consider the following message: "{input_text}". Your task is {task}. {answer_template_specification}"""
    
    if strategy == "role_based_one_shot":
        user_msg = f"""{model_instruction}. {task_instruction}. Your task is {task}. {answer_template_specification}.
                        Consider the following example messages:
                        Example 1: "{civil_example1}". \
                        Response for Example 1: {{"label":"CIVIL"}}
                        Example 2: {uncivil_example1} \
                        Response for Example 2: {{"label":"UNCIVIL"}}
                        Now, analyze the following message:
                        "{input_text}" """
        

    if strategy == "role_based_few_shot":
        user_msg = f"""{model_instruction}. {task_instruction}. Your task is {task}. {answer_template_specification}.
                Consider the following example messages:
                Example 1: "{civil_example1}". 
                Response for Example 1: {{"label":"CIVIL"}} 
                Example 2: {civil_example2}  \
                Response for Example 2: {{"label":"CIVIL"}} 
                Example 3: "{civil_example3}". \
                Response for Example 3: {{"label":"CIVIL"}} 
                Example 4: {uncivil_example1} \
                Response for Example 4: {{"label":"UNCIVIL"}}
                Example 5: "{uncivil_example2}". \
                Response for Example 5: {{"label":"UNCIVIL"}}
                Example 6: {uncivil_example3} \
                Response for Example 6: {{"label":"UNCIVIL"}}
                Now, analyze the following message:
                "{input_text}" """
    
    if strategy == "role_based_auto_cot":
        user_msg = f"""{model_instruction}. {task_instruction}. Consider the following message: {input_text}. Your task is {task}. {cot_promotion}. {answer_template_specification_auto_cot}"""

    return {'sys_msg': sys_msg, 'user_msg':user_msg}
        
#strategies = ['zero_shot', 'one_shot', 'few_shot', 'auto_cot', 'role_based', 'role_based_few_shot', 'role_based_auto_cot', 'role_based_one_shot']
strategies = ['zero_shot']