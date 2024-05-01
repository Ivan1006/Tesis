import pandas as pd
import random
import re
from transformers import pipeline
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np



def llm_model(data):

    frecuencias = data['Etiquetas'].value_counts() 
    print (frecuencias)
    data_filtered = data[data['Etiquetas'].apply(lambda x: 100 <= frecuencias[x] <= 300)]
    data_3000 = data_filtered.sample(n=min(300, len(data_filtered)))

    data_3000.reset_index(drop=True, inplace=True)
    indices_a_eliminar = data_3000[~data_3000['JUSTIFICACION'].apply(lambda x: isinstance(x, str))].index
    data_3000 = data_3000.drop(indices_a_eliminar)

    data_3000['Etiquetas'] = list(zip(data_3000['RUBRO'], data_3000['PRODUCTO RELACIONADO']))
    data_3000.drop_duplicates(subset=['NIT','Etiquetas'], inplace=True)

    frecuencias = data_3000['Etiquetas'].value_counts()

    print("Frecuencias antes del filtro:")
    print(frecuencias)

    data_3000 = data_3000[data_3000.apply(lambda fila: frecuencias[fila['Etiquetas']] > 15, axis=1)]
    print("Después del filtro:")
    print(data_3000)

    datos_a_clasificar = data[~data.index.isin(data_3000.index)].sample(n=50)
    datos_a_clasificar['Etiquetas'] = list(zip(datos_a_clasificar['RUBRO'], datos_a_clasificar['PRODUCTO RELACIONADO']))


    clasificador_llm = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

    def clasificar_llm(texto, etiquetas_posibles):
        clasificacion = clasificador_llm(texto, candidate_labels=etiquetas_posibles, multi_label=True)
        return clasificacion['labels'], clasificacion['scores']

    textos_entrenamiento = data_3000['JUSTIFICACION'].tolist()
    etiquetas_entrenamiento = data_3000['Etiquetas'].tolist()

    def preparar_ejemplos(textos, etiquetas):
        ejemplos = [{'text': texto, 'labels': etiqueta} for texto, etiqueta in zip(textos, etiquetas)]
        print(ejemplos)
        return ejemplos


    ejemplos_entrenamiento = preparar_ejemplos(textos_entrenamiento, etiquetas_entrenamiento)

    def clasificar_con_llm(texto):

        if isinstance(texto, str):
            etiqueta_predicha = clasificar_llm(texto, etiquetas_entrenamiento)
            return etiqueta_predicha
        else:
            return None  

    datos_a_clasificar['Etiquetas_LLM'] = datos_a_clasificar['JUSTIFICACION'].apply(clasificar_con_llm)

    # Convertir las etiquetas de entrenamiento a un formato binarizado
    mlb = MultiLabelBinarizer()
    etiquetas_binarizadas = mlb.fit_transform(datos_a_clasificar['Etiquetas'])

    # Clasificar los textos y obtener las probabilidades para cada etiqueta
    probabilidades = []
    for texto in datos_a_clasificar['JUSTIFICACION'].tolist():
        _, scores = clasificar_llm(texto, mlb.classes_)
        probabilidades.append(scores)

    probabilidades = np.array(probabilidades)


    # Asumiendo que `probabilidades` y `etiquetas_binarizadas` están correctamente formateadas
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, clase in enumerate(mlb.classes_):
        fpr, tpr, thresholds = roc_curve(etiquetas_binarizadas[:, i], probabilidades[:, i])
        auc_score = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'Clase {clase} (AUC = {auc_score:.2f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlabel('Tasa de Falsos Positivos')
    ax.set_ylabel('Tasa de Verdaderos Positivos')
    ax.set_title('Curva ROC para cada clase')
    ax.legend(loc="lower right")
    plt.show()
