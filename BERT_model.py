import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
from itertools import cycle

def BERT_model(data):
    # Tokenización
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 256

    def tokenize_function(examples):
        return tokenizer(examples, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    
    # Preparar inputs
    inputs = tokenize_function(data['JUSTIFICACION'].tolist())
    input_ids = inputs['input_ids']
    attention_masks = inputs['attention_mask']

    # Preparación de etiquetas
    mlb = MultiLabelBinarizer()
    # Transformar cada tupla en una cadena única
    # Limpieza y transformación de la columna 'Etiquetas'
    data['Etiquetas'] = data['Etiquetas'].apply(
        lambda x: ['-'.join(map(str, tup)) if isinstance(tup, tuple) else str(x) for tup in x] 
        if isinstance(x, list) else [str(x)]
    )
    print(data['Etiquetas'].value_counts())
    labels = mlb.fit_transform(data['Etiquetas'])
    labels = torch.tensor(labels, dtype=torch.float32)
    print(mlb.classes_)
    print(labels.sum(axis=0))  # Utiliza el método .sum() de PyTorch directamente

    # Crear dataset
    dataset = TensorDataset(input_ids, attention_masks, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    validation_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Modelo BERT
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(mlb.classes_))
    model.cuda()

    # Optimización
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_dataloader) * 6
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)




    # Entrenamiento
    for epoch in range(6):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            batch = tuple(t.to(model.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            model.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        print(f"Epoch {epoch + 1}, Average Training Loss: {total_loss / len(train_dataloader)}")

    # Evaluación
    model.eval()
    all_logits = []
    all_labels = []
    for batch in validation_dataloader:
        batch = tuple(t.to(model.device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs.logits

        all_logits.append(logits.cpu().numpy())
        all_labels.append(b_labels.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    predicted_classes = np.argmax(all_logits, axis=1)
    true_classes = np.argmax(all_labels, axis=1)

    # Matriz de confusión
    cm = confusion_matrix(true_classes, predicted_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=mlb.classes_, yticklabels=mlb.classes_)
    plt.title('Matriz de Confusión para BERT')
    plt.ylabel('Etiquetas Verdaderas')
    plt.xlabel('Etiquetas Predichas')
    plt.show()

    # Informe de clasificación
    report = classification_report(true_classes, predicted_classes, target_names=mlb.classes_)
    print(report)

    # ROC Curve
    probabilities = softmax(all_logits, axis=1)
    plt.figure(figsize=(10, 8))
    for i, color in zip(range(len(mlb.classes_)), cycle(['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightblue', 'lightgreen'])):
        fpr, tpr, _ = roc_curve(all_labels[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, label=f'Class {mlb.classes_[i]} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    return true_classes, predicted_classes, mlb.classes_
