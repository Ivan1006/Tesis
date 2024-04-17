import pandas as pd
import random
from transformers import pipeline
import re
from transformers import pipeline


def limpiar_texto(texto):

    texto = texto.lower()

    texto = re.sub(r'[^\w\s]', ' ', texto)

    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto




data = pd.read_excel('exceles_info/Informacion.xlsx')

print(data['JUSTIFICACION'])

data = data.sample(n=200)

data.reset_index(drop=True, inplace=True)

indices_a_eliminar = data[~data['JUSTIFICACION'].apply(lambda x: isinstance(x, str))].index

data = data.drop(indices_a_eliminar)

data['JUSTIFICACION'] = data['JUSTIFICACION'].apply(limpiar_texto)

data['Etiquetas'] = list(zip(data['RUBRO'], data['PRODUCTO RELACIONADO']))

data.drop_duplicates(subset=['NIT','Etiquetas'],inplace=True)

frecuencias = data['Etiquetas'].value_counts()

data['Etiquetas'] = data.apply(lambda fila: fila['Etiquetas'] if frecuencias[fila['Etiquetas']] > 90 else (0, 0), axis=1)

# Calcula las frecuencias de cada etiqueta
frecuencias = data['Etiquetas'].value_counts().reset_index()
frecuencias.columns = ['Etiquetas', 'Frecuencia']

# Calcula el 10% de cada etiqueta
frecuencias['Muestra'] = (frecuencias['Frecuencia'] * 0.1).round().astype(int)

# Asegura no tener muestras de tamaño cero
frecuencias['Muestra'] = frecuencias['Muestra'].apply(lambda x: 1 if x == 0 else x)

# Crea un DataFrame vacío para los ejemplos
ejemplos_df = pd.DataFrame()

# Muestrea de 'data' para cada etiqueta
for _, row in frecuencias.iterrows():
    muestra = data[data['Etiquetas'] == row['Etiquetas']].sample(n=row['Muestra'])
    ejemplos_df = pd.concat([ejemplos_df, muestra])


prompt_base = "Clasifica el siguiente texto en base a los ejemplos:\n\n"

for _, fila in ejemplos_df.iterrows():
    prompt_base += f"Texto: \"{fila['JUSTIFICACION']}\" Clasificación: {fila['Etiquetas']}\n"
# Cargar un modelo de lenguaje grande de Hugging Face
generator = pipeline('text-generation', model="EleutherAI/gpt-neo-2.7B")
print(ejemplos_df)
# Función para clasificar un nuevo texto utilizando el prompt de few-shot
def clasificar(texto):
    prompt = prompt_base + f"Texto: \"{texto}\" Clasificación:"
    print(prompt)
    outputs = generator(prompt, max_length=50, num_return_sequences=1)
    result = outputs[0]['generated_text'].split("Clasificación:")[-1].strip()  # Extraer solo la clasificación
    return result

# Aplicar la clasificación a los textos que no fueron parte de los ejemplos
datos_para_clasificar = data.drop(ejemplos_df.index)

# Puedes utilizar .apply() para clasificar el resto de los datos, pero ten en cuenta que esto puede ser lento
# Es recomendable probar primero con una pequeña porción del DataFrame
datos_para_clasificar['Clasificacion_Predicha'] = datos_para_clasificar['JUSTIFICACION'].apply(clasificar)

print(datos_para_clasificar[['JUSTIFICACION', 'Clasificacion_Predicha']])
