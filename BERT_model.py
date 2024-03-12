import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.metrics import classification_report


def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

data = pd.read_excel('Informacion.xlsx')
data['JUSTIFICACION'] = data['JUSTIFICACION'].apply(limpiar_texto)

frecuencias = data['PRODUCTO RELACIONADO'].value_counts()
umbral = frecuencias.quantile(0.80) 
print(umbral)
data['PRODUCTO RELACIONADO ACTUALIZADO'] = data['PRODUCTO RELACIONADO'].apply(
    lambda x: 0 if frecuencias[x] <= umbral else x)
print(data['PRODUCTO RELACIONADO ACTUALIZADO'].value_counts())
label_encoder = LabelEncoder()
data['PRODUCTO RELACIONADO ACTUALIZADO'] = label_encoder.fit_transform(data['PRODUCTO RELACIONADO ACTUALIZADO'].astype(str))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 256

def tokenize_function(examples):
    return tokenizer(examples['JUSTIFICACION'], padding="max_length", truncation=True, max_length=max_length)

# ejemplo = {"JUSTIFICACION": "La empresa PRODUCTORA Y COMERCIALIZADORA JGX S.A.S. constituida en el año 2019, cuenta con una amplia trayectoria en el desarrollo de actividades agrícolas mediante el cultivo y producción de plátano, maracuyá y guayaba en una finca ubicada en el municipio de San Lorenzo, Nariño. El cultivo de plátano lo realiza en forma escalonada sembrando por cada hectárea 2.000 plantas, el cultivo de maracuyá también se maneja de manera escalonada la sacando cada dos meses en promedio de 15 toneladas; por ultimo, el cultivo de guayaba lo maneja con ciclos semestrales y saca un promedio de 50 toneladas."}
# tokens = tokenize_function(ejemplo)
# print(tokens)

# def contar_tokens(texto):
#     tokens = tokenizer.tokenize(texto)
#     return len(tokens)

# data['longitud_tokens'] = data['JUSTIFICACION'].apply(contar_tokens)
# media_tokens = data['longitud_tokens'].mean()

# print(f"La longitud media de tokens es: {media_tokens}")


tokenized_inputs = tokenize_function(data.to_dict(orient='list'))
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
print(classification_report(true_labels, predicted_labels, target_names=label_encoder.classes_))

def predecir_justificacion(modelo, tokenizer, justificacion):
    # Limpiamos el texto
    justificacion_limpiada = limpiar_texto(justificacion)
    
    # Tokenizamos
    inputs = tokenizer(justificacion_limpiada, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    
    # Movemos los tensores a GPU si está disponible
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    
    # Realizamos la predicción
    with torch.no_grad():
        outputs = modelo(input_ids, attention_mask=attention_mask)
    
    # Obtenemos los logits
    logits = outputs.logits
    
    # Convertimos los logits a probabilidades
    probabilidades = torch.softmax(logits, dim=1)
    
    # Obtenemos la clase predicha
    prediccion = torch.argmax(probabilidades, dim=1)
    
    # Convertimos la predicción a numpy y luego a su etiqueta original
    prediccion = prediccion.cpu().numpy()[0]  # Asumiendo una sola predicción
    etiqueta_predicha = label_encoder.inverse_transform([prediccion])[0]
    
    return etiqueta_predicha

# Uso del modelo para una nueva justificación
nueva_justificacion = """COSMOLAC S.A.S Somos una empresa constituida desde el año 2013 dedicados a la elaboración y distribución de productos lácteos en polvo de excelente calidad garantizando la inocuidad y la satisfacción de los clientes, nuestros productos son: leche entera fortificada con H y V Fortificada con vitaminas (A y D3) y hierro aminoquelado, lo que incrementa el valor nutricional. Leche entera azucarada mezclada con azúcar pulverizada en una proporción del (1%). Alimento lácteo Producto obtenido a partir de la mezcla balanceada de leches en polvo, maltodextrina, suero lácteo y grasa. Dicha actividad es desarrollada en el municipio de Zipaquirá, Cundinamarca. Para el 2024, se tiene proyectado un valor de compras al sector agropecuario nacional de $182.365 millones de pesos de leche cruda. """
etiqueta_predicha = predecir_justificacion(model, tokenizer, nueva_justificacion)
print(f"La clasificación predicha es: {etiqueta_predicha}")
