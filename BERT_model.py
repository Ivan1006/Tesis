import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import classification_report
from scipy.special import softmax
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle

def BERT_model(data):
    # Procesamiento de etiquetas
    X = data['JUSTIFICACION']
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(data['Etiquetas'])
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 256
    
    label_encoder = LabelEncoder()
    data['Etiquetas'] = label_encoder.fit_transform(data['Etiquetas'].astype(str))

    def tokenizador(examples):
        return tokenizer(examples['JUSTIFICACION'], padding="max_length", truncation=True, max_length=max_length)

    tokenized_inputs = tokenizador(data.to_dict(orient='list'))
    input_ids = torch.tensor(tokenized_inputs['input_ids'])
    attention_masks = torch.tensor(tokenized_inputs['attention_mask'])
    labels = torch.tensor(data['Etiquetas'].values)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=16)
    validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=16)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(mlb.classes_))
    model.cuda()

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 6
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(0, epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            b_input_ids, b_input_mask, b_labels = [t.cuda() for t in batch]
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

    model.eval()
    predictions_bert, true_labels = [], []

    for batch in validation_dataloader:
        batch = tuple(t.cuda() for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions_bert.append(logits)
        true_labels.append(label_ids)

    predictions_bert = np.concatenate(predictions_bert, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    probabilities_bert = softmax(predictions_bert, axis=1)  # Usando scipy.special.softmax


    probabilities = softmax(predictions_bert, axis=1)

    # Inicializamos la figura para la curva ROC
    plt.figure(figsize=(7, 5))

    # Definimos los colores para cada curva
    colors = cycle(['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightblue', 'lightgreen'])

    # Obtenemos el número de clases
    n_classes = len(np.unique(data['Etiquetas']))

    # Iteramos sobre todas las clases
    for i, color in zip(range(n_classes), colors):
        # Calculamos la ROC para la clase i usando un enfoque one-vs-all
        fpr, tpr, _ = roc_curve((true_labels == i).astype(int), probabilities[:, i])
        roc_auc = auc(fpr, tpr)

        # Dibujamos la curva ROC para la clase i
        plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve (class {i}) (area = {roc_auc:.2f})')

    # Dibujamos la línea de la suerte
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve - BERT')
    plt.legend(loc="lower right")
    plt.show()
    return true_labels, probabilities_bert, mlb.classes_


