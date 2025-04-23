import json
from pathlib import Path

data_path = Path('data')
file_path = data_path / 'examples_fg.json'

# Read the JSON file with examples
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    
example_bitter = data["bitter frustration"]
example_impatience = data["impatience"]
example_mocking = data["mocking"]
example_irony = data["irony"]
example_vulgarity = data["vulgarity"]
example_threat = data["threat"]
example_ia = data["identify attack/name calling"]
example_entitlement = data["entitlement"]
example_insulting = data["insulting"]
example_none = data["none"]



def prompt_factory(input_text,strategy):
  sys_msg = ''
  user_msg = ''
  answer_template_specification_auto_cot = """Your response should be only a json in the following format: {{'label': 'Bitter Frustration' OR 'Impatience' OR 'Mocking' OR 'Irony' OR 'Vulgarity' OR 'Identify Attacks/Name Calling' OR 'Insulting' OR 'Entitlement' OR 'Threat' OR 'None', "reasoning": "Your reasoning here"}}"""
      
  task = """to classify the message into one of the specified classes: 
  Bitter Frustration, Impatience, Mocking, Irony, Vulgarity, Threat, Identify Attack/Name Calling, Entitlement, Insulting or None."""
  
  task_instruction = """
    Mocking are those messages that involves ridiculing or making fun of someone in a disrespectful way; Identity Attack/Name Calling are those messages that involves making derogatory comments based on race, religion, gender, sexual orientation, or nationality; Bitter Frustration are those messages that involves strong frustration, displeasure, or annoyance; Impatience are those messages that involves conveys dissatisfaction due to delays; Threat are those messages that involves issuing a warning that implies a negative consequence; Irony are those messages that involves uses language to imply a meaning opposite to the literal one, often sarcastically; Vulgarity are those messages that involves offensive or inappropriate language; Entitlement are those messages that involves expecting special treatment or privileges; Insulting are those messages that involves making derogatory remarks towards another person or project; and None are messages that do not contain any type of aforementioned incivility
  """
  answer_template_specification = f"""Return the result as a JSON with the following format: {{'label': 'Bitter Frustration' OR 'Impatience' OR 'Mocking' OR 'Irony' OR 'Vulgarity' OR 'Identify Attacks/Name Calling' OR 'Insulting' OR 'Entitlement' OR 'Threat' OR 'None'}}
  Do not include any extra text, explanations, or formatting."""
  
  model_instruction = "**You are a GitHub moderator of conversations** that classifies the tone of messages in GitHub discussions as 'Bitter Frustration' OR 'Impatience' OR 'Mocking' OR 'Irony' OR 'Vulgarity' OR 'Identify Attacks/Name Calling' OR 'Insulting' OR 'Entitlement' OR 'Threat' OR 'None'"
  
  cot_promotion = "Let's think step by step. Provide reasoning before giving the response"
  
  if strategy == 'zero_shot':
    user_msg = f"""Consider the following message: {input_text}. Your task is {task}. {answer_template_specification}
"""
  if strategy == 'one_shot':
    user_msg = f"""Your task is {task}. {answer_template_specification}.
    Consider the following example messages: 
    Example 1: {example_bitter[0]}.
    Response for Example 1: {{'label': 'Bitter Frustration'}} \
    
    Example 2: {example_impatience[0]}.
    Response for Example 2: {{'label': 'Impatience'}} \
      
    Example 3: {example_mocking[0]}.
    Response for Example 3: {{'label': 'Mocking'}} \
    
    Example 4: {example_irony[0]}.
    Response for Example 4: {{'label': 'Irony'}} \
      
    Example 5: {example_vulgarity[0]}.
    Response for Example 5: {{'label': 'Vulgarity'}} \
    
    Example 6: {example_threat[0]}.
    Response for Example 6: {{'label': 'Threat'}} \
      
    Example 7: {example_ia[0]}.
    Response for Example 7: {{'label': 'Identify Attack/Name Calling'}} \
      
    Example 8: {example_entitlement[0]}.
    Response for Example 8: {{'label': 'Entitlement'}} \
      
    Example 9: {example_insulting[0]}.
    Response for Example 9: {{'label': 'Insulting'}} \
      
    Example 10: {example_none[0]}.
    Response for Example 10: {{'label': 'None'}} \
      
    Now, analyze the following message:
    "{input_text}" 
"""
  if strategy == 'few_shot_3':
    user_msg = f"""Your task is {task}. {answer_template_specification}.
    Consider the following example messages: 
    Example 1: {example_bitter[0]}.
    Response for Example 1: {{'label': 'Bitter Frustration'}} \
      
    Example 2: {example_bitter[1]}.
    Response for Example 2: {{'label': 'Bitter Frustration'}} \
      
    Example 3: {example_bitter[2]}.
    Response for Example 3: {{'label': 'Bitter Frustration'}} \
    
    Example 4: {example_impatience[0]}.
    Response for Example 4: {{'label': 'Impatience'}} \
      
    Example 5: {example_impatience[1]}.
    Response for Example 5: {{'label': 'Impatience'}} \
    
    Example 6: {example_impatience[2]}.
    Response for Example 6: {{'label': 'Impatience'}} \
      
    Example 7: {example_mocking[0]}.
    Response for Example 7: {{'label': 'Mocking'}} \
    
    Example 8: {example_mocking[1]}.
    Response for Example 8: {{'label': 'Mocking'}} \
      
    Example 9: {example_mocking[2]}.
    Response for Example 9: {{'label': 'Mocking'}} \
      
    Example 10: {example_irony[0]}.
    Response for Example 10: {{'label': 'Irony'}} \
      
    Example 11: {example_irony[1]}.
    Response for Example 11: {{'label': 'Irony'}} \
      
    Example 12: {example_irony[2]}.
    Response for Example 12: {{'label': 'Irony'}} \
      
    Example 13: {example_vulgarity[0]}.
    Response for Example 13: {{'label': 'Vulgarity'}} \
      
    Example 14: {example_vulgarity[1]}.
    Response for Example 14: {{'label': 'Vulgarity'}} \
      
    Example 15: {example_vulgarity[2]}.
    Response for Example 15: {{'label': 'Vulgarity'}} \
    
    Example 16: {example_threat[0]}.
    Response for Example 16: {{'label': 'Threat'}} \
      
    Example 17: {example_threat[1]}.
    Response for Example 17: {{'label': 'Threat'}} \
      
    Example 18: {example_threat[2]}.
    Response for Example 18: {{'label': 'Threat'}} \
      
    Example 19: {example_ia[0]}.
    Response for Example 19: {{'label': 'Identify Attack/Name Calling'}} \
    
    Example 20: {example_ia[1]}.
    Response for Example 20: {{'label': 'Identify Attack/Name Calling'}} \
    
    Example 21: {example_ia[2]}.
    Response for Example 21: {{'label': 'Identify Attack/Name Calling'}} \
      
    Example 22: {example_entitlement[0]}.
    Response for Example 22: {{'label': 'Entitlement'}} \
      
    Example 23: {example_entitlement[1]}.
    Response for Example 23: {{'label': 'Entitlement'}} \
      
    Example 24: {example_entitlement[2]}.
    Response for Example 24: {{'label': 'Entitlement'}} \
      
    Example 25: {example_insulting[0]}.
    Response for Example 25: {{'label': 'Insulting'}} \
      
    Example 26: {example_insulting[1]}.
    Response for Example 26: {{'label': 'Insulting'}} \
      
    Example 27: {example_insulting[2]}.
    Response for Example 27: {{'label': 'Insulting'}} \
      
    Example 28: {example_none[0]}.
    Response for Example 28: {{'label': 'None'}} \
      
    Example 29: {example_none[1]}.
    Response for Example 29: {{'label': 'None'}} \
      
    Example 30: {example_none[2]}.
    Response for Example 30: {{'label': 'None'}} \
      
    Now, analyze the following message:
    "{input_text}" 
"""
  if strategy == 'auto_cot':
    user_msg = f"""Consider the following message: "{input_text}". Your task is {task}. {cot_promotion}. {answer_template_specification_auto_cot}"""
  
  if strategy == 'role_based':
    user_msg = f"""{model_instruction}. {task_instruction}. Consider the following message: "{input_text}". Your task is {task}. {answer_template_specification}"""
  
  
  if strategy == 'role_based_one_shot':
    user_msg = f"""{model_instruction}. {task_instruction}. Your task is {task}. {answer_template_specification}
    
    Consider the following example messages: 
    Example 1: {example_bitter[0]}.
    Response for Example 1: {{'label': 'Bitter Frustration'}} \
    
    Example 2: {example_impatience[0]}.
    Response for Example 2: {{'label': 'Impatience'}} \
      
    Example 3: {example_mocking[0]}.
    Response for Example 3: {{'label': 'Mocking'}} \
    
    Example 4: {example_irony[0]}.
    Response for Example 4: {{'label': 'Irony'}} \
      
    Example 5: {example_vulgarity[0]}.
    Response for Example 5: {{'label': 'Vulgarity'}} \
    
    Example 6: {example_threat[0]}.
    Response for Example 6: {{'label': 'Threat'}} \
      
    Example 7: {example_ia[0]}.
    Response for Example 7: {{'label': 'Identify Attack/Name Calling'}} \
      
    Example 8: {example_entitlement[0]}.
    Response for Example 8: {{'label': 'Entitlement'}} \
      
    Example 9: {example_insulting[0]}.
    Response for Example 9: {{'label': 'Insulting'}} \
      
    Example 10: {example_none[0]}.
    Response for Example 10: {{'label': 'None'}} \
      
    Now, analyze the following message:
    "{input_text}" 
    """
  if strategy == 'role_based_few_shot_3':
    user_msg = f"""{model_instruction}. {task_instruction}. Your task is {task}. {answer_template_specification}
    Consider the following example messages: 
    Example 1: {example_bitter[0]}.
    Response for Example 1: {{'label': 'Bitter Frustration'}} \
      
    Example 2: {example_bitter[1]}.
    Response for Example 2: {{'label': 'Bitter Frustration'}} \
      
    Example 3: {example_bitter[2]}.
    Response for Example 3: {{'label': 'Bitter Frustration'}} \
    
    Example 4: {example_impatience[0]}.
    Response for Example 4: {{'label': 'Impatience'}} \
      
    Example 5: {example_impatience[1]}.
    Response for Example 5: {{'label': 'Impatience'}} \
    
    Example 6: {example_impatience[2]}.
    Response for Example 6: {{'label': 'Impatience'}} \
      
    Example 7: {example_mocking[0]}.
    Response for Example 7: {{'label': 'Mocking'}} \
    
    Example 8: {example_mocking[1]}.
    Response for Example 8: {{'label': 'Mocking'}} \
      
    Example 9: {example_mocking[2]}.
    Response for Example 9: {{'label': 'Mocking'}} \
      
    Example 10: {example_irony[0]}.
    Response for Example 10: {{'label': 'Irony'}} \
      
    Example 11: {example_irony[1]}.
    Response for Example 11: {{'label': 'Irony'}} \
      
    Example 12: {example_irony[2]}.
    Response for Example 12: {{'label': 'Irony'}} \
      
    Example 13: {example_vulgarity[0]}.
    Response for Example 13: {{'label': 'Vulgarity'}} \
      
    Example 14: {example_vulgarity[1]}.
    Response for Example 14: {{'label': 'Vulgarity'}} \
      
    Example 15: {example_vulgarity[2]}.
    Response for Example 15: {{'label': 'Vulgarity'}} \
    
    Example 16: {example_threat[0]}.
    Response for Example 16: {{'label': 'Threat'}} \
      
    Example 17: {example_threat[1]}.
    Response for Example 17: {{'label': 'Threat'}} \
      
    Example 18: {example_threat[2]}.
    Response for Example 18: {{'label': 'Threat'}} \
      
    Example 19: {example_ia[0]}.
    Response for Example 19: {{'label': 'Identify Attack/Name Calling'}} \
    
    Example 20: {example_ia[1]}.
    Response for Example 20: {{'label': 'Identify Attack/Name Calling'}} \
    
    Example 21: {example_ia[2]}.
    Response for Example 21: {{'label': 'Identify Attack/Name Calling'}} \
      
    Example 22: {example_entitlement[0]}.
    Response for Example 22: {{'label': 'Entitlement'}} \
      
    Example 23: {example_entitlement[1]}.
    Response for Example 23: {{'label': 'Entitlement'}} \
      
    Example 24: {example_entitlement[2]}.
    Response for Example 24: {{'label': 'Entitlement'}} \
      
    Example 25: {example_insulting[0]}.
    Response for Example 25: {{'label': 'Insulting'}} \
      
    Example 26: {example_insulting[1]}.
    Response for Example 26: {{'label': 'Insulting'}} \
      
    Example 27: {example_insulting[2]}.
    Response for Example 27: {{'label': 'Insulting'}} \
      
    Example 28: {example_none[0]}.
    Response for Example 28: {{'label': 'None'}} \
      
    Example 29: {example_none[1]}.
    Response for Example 29: {{'label': 'None'}} \
      
    Example 30: {example_none[2]}.
    Response for Example 30: {{'label': 'None'}} \
      
    Now, analyze the following message:
    "{input_text}" 
"""
  if strategy == 'role_based_auto_cot':
    user_msg = f"""{model_instruction}. {task_instruction}. Consider the following message: {input_text}. Your task is {task}. {cot_promotion}. {answer_template_specification_auto_cot}"""
  
  return {'sys_msg': sys_msg, 'user_msg': user_msg}
