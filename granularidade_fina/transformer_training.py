import torch
import pandas as pd
import numpy as np
import os 
from pathlib import Path
import json

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

#Bert/Model general config
model_name = 'distilbert-base-uncased'
device_name = 'cuda'
max_length = 512
cached_model_directory_name = 'distilbert-incivility'

# Data Loading and basic processing
dataset = pd.read_csv(Path(r'data/reference_dataset_fg.csv'))

tokenizer = DistilBertTokenizerFast.from_pretrained(model_name) # The model_name needs to match our pre-trained model.

encoder = {k:v for v, k in enumerate(dataset['tbdf'].unique())}
print("encoder:", encoder)

dataset['tbdf_label'] = dataset['tbdf'].replace(encoder)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
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
    preds = np.argmax(pred.predictions, axis=1)

    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='weighted', zero_division=0),
        'recall': recall_score(labels, preds, average='weighted', zero_division=0),
        'f1': f1_score(labels, preds, average='weighted', zero_division=0),
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
print("Primeiros elementos de dataset['tbdf_label']:", dataset['tbdf_label'].iloc[:5])

# Verificar o tipo dos rótulos
print("Tipo dos rótulos:", type(dataset['tbdf_label'].iloc[0]))

# Definindo mapeamento de rótulos
# Adaptando para aceitar tanto strings quanto inteiros/floats
#  label_mapping = {'incivility': 1, 'civility': 0, 1: 1, 0: 0, '1': 1, '0': 0}
label_mapping = {'none': 0, 'entitlement': 1, 'impatience': 2,
                 'bitter frustration': 3, 'mocking': 4,
                 'irony': 5, 'vulgarity': 6,
                 'identify attack/name calling': 7,
                 'insulting': 8, 'threat': 9,
                 '0': 0, '1': 1, '2': 2, '3': 3,
                 '4': 4, '5': 5, '6': 6, '7': 7, '8':8, '9':9,
                  0:0, 1: 1, 2: 2, 3: 3,
                  4: 4, 5: 5, 6: 6, 7: 7, 8:8, 9:9}

num_classes = len(dataset['tbdf'].unique())

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=51)
X, y = dataset['comment_body'], dataset['tbdf_label']
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
            return int(label_mapping[label])
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
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes).to(device_name)
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
    probas = predicted_results.predictions
    preds = np.argmax(probas, axis=1)

    # Calculando e armazenando métricas
    folds[i] = {}
    folds[i]['pre'] = precision_score(test_labels_encoded, preds, average='weighted', zero_division=0)
    folds[i]['rec'] = recall_score(test_labels_encoded, preds, average='weighted', zero_division=0)
    folds[i]['acc'] = accuracy_score(test_labels_encoded, preds)
    folds[i]['f1'] = f1_score(test_labels_encoded, preds, average='weighted', zero_division=0)

    # AUC multiclasse (usando one-vs-rest)
    try:
        y_true_bin = label_binarize(test_labels_encoded, classes=list(range(num_classes)))
        folds[i]['auc'] = roc_auc_score(y_true_bin, probas, average='weighted', multi_class='ovr')
    except Exception as e:
        print("Erro ao calcular AUC:", e)
        folds[i]['auc'] = 0.0
    
    # Salvando as previsões para posterior análise se necessário
    report = classification_report(test_labels_encoded, preds, output_dict=True, zero_division=0)
    for cls in range(num_classes):
        cls_str = str(cls)
        folds[i][f'f1_cls_{cls_str}'] = report[cls_str]['f1-score']
        folds[i][f'pre_cls_{cls_str}'] = report[cls_str]['precision']
        folds[i][f'rec_cls_{cls_str}'] = report[cls_str]['recall']
        folds[i][f'support_cls_{cls_str}'] = report[cls_str]['support']

    # Salvando previsões e probabilidades
    folds[i]['predictions'] = preds.tolist()

    results_transformers_probs = results_transformers / 'probas'
    results_transformers_probs.mkdir(parents=True, exist_ok=True)
    np.save(results_transformers_probs / f'fold_{i}_probas.npy', probas)
    folds[i]['probabilities'] = f'fold_{i}_probas.npy'

    print(f"Fold {i+1}=> PRE: {folds[i]['pre']:.4f}; REC: {folds[i]['rec']:.4f}; ACC: {folds[i]['acc']:.4f}; F1S: {folds[i]['f1']:.4f}; AUC: {folds[i]['auc']:.4f}")


# Salvando os resultados em um arquivo JSON
def convert_ndarrays(obj):
    if isinstance(obj, dict):
        return {k: convert_ndarrays(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarrays(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

folds_serializable = convert_ndarrays(folds)

with open(results_transformers / "DistilBert-10folds_results.json", "w") as fp:
    json.dump(folds_serializable, fp, indent=2)


# Calculando e imprimindo métricas médias
metrics = ['pre', 'rec', 'acc', 'f1', 'auc']
avg_metrics = {metric: sum(folds[i][metric] for i in folds) / len(folds) for metric in metrics}

print("\nMétricas médias dos 10 folds:")
for metric, value in avg_metrics.items():
    print(f"{metric.upper()}: {value:.4f}")