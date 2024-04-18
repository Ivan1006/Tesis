import pandas as pd
import re
from RF_model import RF_model


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
    data['Etiquetas'] = data.apply(lambda fila: fila['Etiquetas'] if frecuencias[fila['Etiquetas']] > 150 else (0, 0), axis=1)

    return data


def main():

    data = load_data()
    RF_model(data)

if __name__ == '__main__':
    main()