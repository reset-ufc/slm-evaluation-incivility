import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from tqdm import tqdm

# Configurando dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregando tokenizer e modelo
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2, output_attentions=False, output_hidden_states=False
)
model.to(device)

# Carregando pesos do modelo treinado
model_path = Path('model/MyModel_BERT.model')
model.load_state_dict(torch.load(model_path, map_location=device))

# Carregando dataset
data_path = Path('data/final_df.csv')
df = pd.read_csv(data_path)

# Função para testar o modelo
def test_model(dataloader_test):
    model.eval()
    predictions = []

    # Barra de progresso
    for batch in tqdm(dataloader_test, desc="Inference", unit="batch"):
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'labels': batch[2],
        }
        
        with torch.no_grad():        
            outputs = model(**inputs)
        
        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)

    predictions = np.concatenate(predictions, axis=0)
    return predictions

# Função principal para detecção
def detection(df):
    encoded_test_val = tokenizer.batch_encode_plus(
        df.message.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=512,
        return_tensors='pt'
    )

    input_ids_test = encoded_test_val['input_ids']
    attention_masks_test = encoded_test_val['attention_mask']
    labels_test = torch.tensor(df.actual.values)

    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)
    batch_size = 8
    dataloader_test = DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=batch_size)

    pred_test = test_model(dataloader_test)
    preds_flat_test = np.argmax(pred_test, axis=1).flatten()
    return preds_flat_test

# Inferência e geração de resultados
pred_by_refined_model = detection(df)

pred_by_refined_model_df = pd.DataFrame()
pred_by_refined_model_df['message'] = df['message']
pred_by_refined_model_df['pred_by_refined_model'] = pred_by_refined_model
pred_by_refined_model_df['actual'] = df['actual']
pred_by_refined_model_df['source'] = df['source']

print(pred_by_refined_model_df.head())

# Salvando previsões em CSV
results_dir = Path('results')
refined_model_path = Path('refined_model')
refined_model_path.mkdir(parents=True, exist_ok=True)
pred_by_refined_model_df.to_csv(refined_model_path / 'pred_by_refined_model.csv', index=False)

# Relatório de classificação
print(classification_report(df['actual'], pred_by_refined_model))
report_df = pd.DataFrame(classification_report(df['actual'], pred_by_refined_model, output_dict=True)).transpose()
report_df.to_csv(refined_model_path / 'report_by_refined_model.csv', index=False)

# Matriz de confusão e heatmap
conf_matrix = confusion_matrix(df['actual'], pred_by_refined_model)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['CIVIL', 'UNCIVIL'], yticklabels=['CIVIL', 'UNCIVIL'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(refined_model_path / 'cm_refined_model.png')

# Salvar a matriz de confusão como csv
conf_matrix_df = pd.DataFrame(conf_matrix, index=['CIVIL', 'UNCIVIL'], columns=['CIVIL', 'UNCIVIL'])
conf_matrix_df.to_csv(refined_model_path / 'cm_df_refined_model.csv')