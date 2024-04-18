import pandas as pd
import random
import re
from transformers import pipeline

# Función para limpiar el texto
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# Cargar los datos desde el archivo Excel
data = pd.read_excel('exceles_info/Informacion.xlsx')
data['Etiquetas'] = list(zip(data['RUBRO'], data['PRODUCTO RELACIONADO']))
data.drop_duplicates(subset=['NIT','Etiquetas'],inplace=True)

# Seleccionar 3000 datos aleatorios y limpiar las justificaciones
frecuencias = data['Etiquetas'].value_counts() 
print (frecuencias)
data_filtered = data[data['Etiquetas'].apply(lambda x: 100 <= frecuencias[x] <= 300)]
data_3000 = data_filtered.sample(n=min(300, len(data_filtered)))

data_3000.reset_index(drop=True, inplace=True)
indices_a_eliminar = data_3000[~data_3000['JUSTIFICACION'].apply(lambda x: isinstance(x, str))].index
data_3000 = data_3000.drop(indices_a_eliminar)
data_3000['JUSTIFICACION'] = data_3000['JUSTIFICACION'].apply(limpiar_texto)

# Crear etiquetas combinando 'RUBRO' y 'PRODUCTO RELACIONADO'
data_3000['Etiquetas'] = list(zip(data_3000['RUBRO'], data_3000['PRODUCTO RELACIONADO']))
data_3000.drop_duplicates(subset=['NIT','Etiquetas'], inplace=True)

# Contar las frecuencias de las etiquetas
frecuencias = data_3000['Etiquetas'].value_counts()

# Imprimir las frecuencias antes del filtro
print("Frecuencias antes del filtro:")
print(frecuencias)

# Filtrar etiquetas basado en frecuencia
data_3000 = data_3000[data_3000.apply(lambda fila: frecuencias[fila['Etiquetas']] > 15, axis=1)]
# Imprimir el DataFrame después del filtro
print("Después del filtro:")
print(data_3000)

# Seleccionar los 200 datos para clasificar
datos_a_clasificar = data[~data.index.isin(data_3000.index)].sample(n=100)
datos_a_clasificar['Etiquetas'] = list(zip(datos_a_clasificar['RUBRO'], datos_a_clasificar['PRODUCTO RELACIONADO']))


# Crear un pipeline de clasificación de texto con GPT-3
clasificador_llm = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

# Función para clasificar texto utilizando few-shot learning con GPT-3
def clasificar_llm(texto, etiquetas_posibles):
    # Clasificar el texto utilizando few-shot learning
    clasificacion = clasificador_llm(texto, etiquetas_posibles)
    # Devolver la etiqueta predicha
    return clasificacion['labels'][0]

# Datos de entrenamiento
textos_entrenamiento = data_3000['JUSTIFICACION'].tolist()
etiquetas_entrenamiento = data_3000['Etiquetas'].tolist()

# Función para preparar los ejemplos para few-shot learning
def preparar_ejemplos(textos, etiquetas):
    ejemplos = [{'text': texto, 'labels': etiqueta} for texto, etiqueta in zip(textos, etiquetas)]
    print(ejemplos)
    return ejemplos


# Preparar los ejemplos para few-shot learning
ejemplos_entrenamiento = preparar_ejemplos(textos_entrenamiento, etiquetas_entrenamiento)

# Función para clasificar texto de prueba utilizando few-shot learning con GPT-3
def clasificar_con_llm(texto):
    # Verificar si el texto es válido
    if isinstance(texto, str):
        # Clasificar el texto utilizando few-shot learning
        etiqueta_predicha = clasificar_llm(texto, etiquetas_entrenamiento)
        return etiqueta_predicha
    else:
        return None  # O manejar el caso de texto no válido de otra manera

# Aplicar la función de clasificación a los datos de prueba
datos_a_clasificar['Etiquetas_LLM'] = datos_a_clasificar['JUSTIFICACION'].apply(clasificar_con_llm)

# Imprimir los resultados de clasificación con few-shot learning
print(datos_a_clasificar[['JUSTIFICACION', 'Etiquetas_LLM']])
