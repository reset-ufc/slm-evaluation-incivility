import torch
import pandas as pd
import numpy as np
import os 
from pathlib import Path
import json

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

#Bert/Model general config
model_name = 'distilbert-base-uncased'
device_name = 'cuda'
max_length = 512
cached_model_directory_name = 'distilbert-incivility'

# Data Loading and basic processing
dataset = pd.read_csv(r'./data/final_df_cleaned.csv')


tokenizer = DistilBertTokenizerFast.from_pretrained(model_name) # The model_name needs to match our pre-trained model.

dataset['actual_label'] = dataset['actual'].replace({'incivility': 1, 'civility': 0})

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def cap_number(x):
    if x > 1:
      return 1
    elif x < 0:
      return 0
    else:
      return x

def compute_metrics(pred):
    labels = pred.label_ids
    outputs = pred.predictions.flatten().tolist()
    probas = [cap_number(x) for x in outputs]
    preds = np.array(np.array(probas) > 0.5, dtype=int)
    acc = accuracy_score(labels, preds)
    return {
      'accuracy': acc,
    }

results_path = Path('./results')
results_transformers = results_path / 'distil_bert_results'
results_transformers.mkdir(parents=True, exist_ok=True)

logs_results = results_transformers / 'logs'


training_args = TrainingArguments(
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=20,   # batch size for evaluation
    learning_rate=5e-5,              # initial learning rate for Adam optimizer
    warmup_steps=100,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    output_dir=results_transformers,          # output directory
    logging_dir=logs_results,            # directory for storing logs
    logging_steps=150,               # number of steps to log
    eval_steps=150,                  # evaluate every 150 steps
    do_eval=True,                    # perform evaluation
    save_steps=150,                  # save model every 150 steps
)

# Imprimindo os primeiros elementos para debug
print("Primeiros elementos de dataset['actual']:", dataset['actual'].iloc[:5])

# Verificar o tipo dos rótulos
print("Tipo dos rótulos:", type(dataset['actual'].iloc[0]))

# Definindo mapeamento de rótulos
# Adaptando para aceitar tanto strings quanto inteiros/floats
label_mapping = {'incivility': 1, 'civility': 0, 1: 1, 0: 0, '1': 1, '0': 0}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=51)
X, y = dataset['message'], dataset['actual']
folds = {}

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"Fold {i+1}: Train Size {len(train_index)} | Test Size {len(test_index)}")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Convertendo para string caso não seja
    X_train = [str(i) for i in X_train]
    X_test = [str(i) for i in X_test]

    # Tokenização dos dados
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=max_length)

    # Função segura para converter rótulos
    def safe_convert_label(label):
        # Tenta converter para float se for uma string numérica
        if isinstance(label, str) and label.isdigit():
            label = int(label)
            
        # Verifica se o rótulo está no dicionário
        if label in label_mapping:
            return float(label_mapping[label])
        else:
            print(f"Rótulo inesperado encontrado: {label}, tipo: {type(label)}")
            # Retorna 0 como valor padrão ou levanta uma exceção
            return 0.0

    # Codificando os rótulos com tratamento de erro
    train_labels_encoded = [safe_convert_label(yi) for yi in y_train]
    test_labels_encoded = [safe_convert_label(yi) for yi in y_test]

    # Criando datasets
    train_dataset = MyDataset(train_encodings, train_labels_encoded)
    test_dataset = MyDataset(test_encodings, test_labels_encoded)

    # Criando e treinando o modelo
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device_name)
    trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=test_dataset,
      compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.evaluate()
    
    # Fazendo previsões e salvando métricas
    predicted_results = trainer.predict(test_dataset)
    outputs = predicted_results.predictions.flatten().tolist()
    probas = [cap_number(x) for x in outputs]
    preds = np.array(np.array(probas) > 0.5, dtype=int)

    # Calculando e armazenando métricas
    folds[i] = {}
    folds[i]['pre'] = precision_score(test_labels_encoded, preds)
    folds[i]['rec'] = recall_score(test_labels_encoded, preds)
    folds[i]['acc'] = accuracy_score(test_labels_encoded, preds)
    folds[i]['auc'] = roc_auc_score(test_labels_encoded, probas)
    folds[i]['f1'] = f1_score(test_labels_encoded, preds)
    
    # Salvando as previsões para posterior análise se necessário
    folds[i]['predictions'] = preds.tolist()
    folds[i]['probabilities'] = probas

    print(f"Fold {i+1}=> PRE: {folds[i]['pre']:.4f}; REC: {folds[i]['rec']:.4f}; ACC: {folds[i]['acc']:.4f}; F1S: {folds[i]['f1']:.4f}; AUC: {folds[i]['auc']:.4f}")


# Salvando os resultados em um arquivo JSON
with open(results_transformers / 'DistilBert-5folds_results.json', 'w') as fp:
    json.dump(folds, fp)

# Calculando e imprimindo métricas médias
metrics = ['pre', 'rec', 'acc', 'f1', 'auc']
avg_metrics = {metric: sum(folds[i][metric] for i in folds) / len(folds) for metric in metrics}

print("\nMétricas médias dos 5 folds:")
for metric, value in avg_metrics.items():
    print(f"{metric.upper()}: {value:.4f}")
