import pandas as pd
import re
from RF_model import RF_model
from BERT_model import BERT_model
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from Llm import llm_model


def limpiar_texto(texto):

    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def load_data():

    data = pd.read_excel('exceles_info/Informacion.xlsx')
    indices_a_eliminar = data[~data['JUSTIFICACION'].apply(lambda x: isinstance(x, str))].index
    data = data.drop(indices_a_eliminar)
    data['JUSTIFICACION'] = data['JUSTIFICACION'].apply(limpiar_texto)
    data['Etiquetas'] = list(zip(data['RUBRO'], data['PRODUCTO RELACIONADO']))
    data.drop_duplicates(subset=['NIT','Etiquetas'],inplace=True)
    frecuencias = data['Etiquetas'].value_counts()
    data['Etiquetas'] = data.apply(lambda fila: fila['Etiquetas'] if frecuencias[fila['Etiquetas']] > 100 else (0, 0), axis=1)
    print(data['Etiquetas'].value_counts())

    return data

def main():
    data = load_data()

    # RF model
    #y_test_rf, probabilities_rf, classes_rf = RF_model(data)
    true_labels_bert, probabilities_bert, classes_bert = BERT_model(data)
    #llm_model(data)

if __name__ == '__main__':
    main()
