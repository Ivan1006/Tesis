import pandas as pd
import numpy as np
import re
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from time import time
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy.special import softmax
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from torch.optim import AdamW 
import pandas  as pd

def limpiar_texto(texto):

    texto = texto.lower()

    texto = re.sub(r'[^\w\s]', ' ', texto)

    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto


data = pd.read_excel('exceles_info/Informacion.xlsx')

data['JUSTIFICACION'] = data['JUSTIFICACION'].apply(limpiar_texto)

frecuencias = data['PRODUCTO RELACIONADO'].value_counts()
umbral = frecuencias.quantile(0.80) 

data['PRODUCTO RELACIONADO ACTUALIZADO'] = data['PRODUCTO RELACIONADO'].apply(
    lambda x: 0 if frecuencias[x] <= umbral else x)


X = data['JUSTIFICACION']
#y = data['PRODUCTO RELACIONADO']
y = data['PRODUCTO RELACIONADO ACTUALIZADO']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo de Random forest
inicio_rf = time()
mejores_parametros_rf = {
    'n_estimators': 400,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt'
}


pipeline_rf_optimizado = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(random_state=42, **mejores_parametros_rf))
])


pipeline_rf_optimizado.fit(X_train, y_train)

predictions = pipeline_rf_optimizado.predict(X_test)
df_rf =(classification_report(y_test, predictions,output_dict=True))
df_rf = pd.DataFrame(df_rf).transpose
fin_rf = time()
tiempo_rf = fin_rf - inicio_rf

# Modelo de BERT
inicio_bert = time()
label_encoder = LabelEncoder()
data['PRODUCTO RELACIONADO ACTUALIZADO'] = label_encoder.fit_transform(data['PRODUCTO RELACIONADO ACTUALIZADO'].astype(str))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 256

def tokenizador(examples):
    return tokenizer(examples['JUSTIFICACION'], padding="max_length", truncation=True, max_length=max_length)

tokenized_inputs = tokenizador(data.to_dict(orient='list'))
input_ids = torch.tensor(tokenized_inputs['input_ids'])
attention_masks = torch.tensor(tokenized_inputs['attention_mask'])
labels = torch.tensor(data['PRODUCTO RELACIONADO ACTUALIZADO'].values)

dataset = TensorDataset(input_ids, attention_masks, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=16)
validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=16)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))
model.cuda()

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 6
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

for epoch in range(0, epochs):
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):

        b_input_ids = batch[0].cuda()
        b_input_mask = batch[1].cuda()
        b_labels = batch[2].cuda()
        b_labels = b_labels.long()
        

        model.zero_grad()        

        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}, Average Training Loss: {avg_train_loss}")

# Evaluación
    
model.eval()
predictions, true_labels = [], []

for batch in validation_dataloader:
    batch = tuple(t.cuda() for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    logits = outputs.logits
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    predictions.append(logits)
    true_labels.append(label_ids)

predictions = np.concatenate(predictions, axis=0)
true_labels = np.concatenate(true_labels, axis=0)
predicted_labels = np.argmax(predictions, axis=1)
print(len(label_encoder.classes_))
print(np.unique(true_labels))

df_bert = classification_report(true_labels, predicted_labels, target_names=label_encoder.classes_,output_dict=True)
df_bert = pd.DataFrame(df_bert).transpose
print(type(df_rf))
print(type(df_bert))

fin_bert = time()
tiempo_bert = fin_bert - inicio_bert

y_test_binarized = label_binarize(y_test, classes=np.unique(y_train))

y_pred_proba_rf_multi = pipeline_rf_optimizado.predict_proba(X_test)

colors = cycle(['blue', 'red', 'green', 'purple', 'orange'])
n_classes = y_test_binarized.shape[1]

plt.figure(figsize=(7, 5))
for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_proba_rf_multi[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'ROC curve of class {i} (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curve - Random Forest')
plt.legend(loc="lower right")
plt.show()

y_pred_proba_bert = softmax(predictions, axis=1)

y_test_binarized = label_binarize(true_labels, classes=np.unique(true_labels))

colors = cycle(['blue', 'red', 'green', 'purple', 'orange'])
n_classes = y_test_binarized.shape[1]

plt.figure(figsize=(7, 5))
for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_proba_bert[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'ROC curve of class {i} (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curve - BERT')
plt.legend(loc="lower right")
plt.show()

#df_rf.to_excel("exceles_info/classification_report_model1.xlsx")
#df_bert.to_excel("exceles_info/classification_report_model2.xlsx")


print(f"Tiempo Random Forest: {tiempo_rf}")
print(f"Tiempo BERT: {tiempo_bert}")
