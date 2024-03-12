import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import re
from openpyxl import load_workbook
from PyPDF2 import PdfReader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD

# def extraer_texto_pdf(ruta_archivo):
#     with open(ruta_archivo, 'rb') as archivo:
#         lector_pdf = PdfReader(archivo)
#         texto_total = ""
#         for pagina in range(len(lector_pdf.pages)):
#             texto_pagina = lector_pdf.pages[pagina].extract_text()  
#             texto_total += texto_pagina
#         return texto_total

# def extraer_informacion(texto):
#     patrones = {
#         "NIT": r"(\d{9}-\d{1})", 
#         "Razón social": r"(?:NOMBRE|RAZÓN SOCIAL|Razón social|Nombre):\s*([\s\S]*?)(?=\n\S)",  
#         "Correo Electrónico": r"[\w.-]+@[\w.-]+.\w+",  
#         "Teléfono": r"(\b\d{10}\b|\b\d{7}\b)",  
#         "Municipio": r"(?:MUNICIPIO|DOMICILIO|ADMINISTRACIÓN DIAN|Municipio):\s*([\w\s]+)",  
#         "Código CIIU": r"(?:Actividad principal Código CIIU|ACTIVIDAD PRINCIPAL|Código ciiu|Código CIIU)\s*:\s*([A-Za-z]?\d{4})",  
#     }
#     resultados = {}
#     for clave, patron in patrones.items():
#         coincidencias = re.findall(patron, texto)
#         if coincidencias:
#             valor = coincidencias[0]
#             resultados[clave] = valor.strip() if clave != "Razón social" else valor
#         else:
#             resultados[clave] = "No encontrado"

#     return resultados

# texto_pdf = extraer_texto_pdf(r"D:\Tesis maestria\Langchain\pdfs\camara de comercio (2).pdf")
# informacion = extraer_informacion(texto_pdf)


def limpiar_texto(texto):

    texto = texto.lower()

    texto = re.sub(r'[^\w\s]', ' ', texto)

    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto


data = pd.read_excel('exceles_info/Informacion.xlsx')

indices_a_eliminar = data[~data['JUSTIFICACION'].apply(lambda x: isinstance(x, str))].index

data = data.drop(indices_a_eliminar)

data['JUSTIFICACION'] = data['JUSTIFICACION'].apply(limpiar_texto)

# data['RUBRO ACTUALIZADO'] = data.apply(lambda fila: fila['RUBRO'] if frecuencias[fila['RUBRO']] > umbral else 0, axis=1)

# data['PRODUCTO RELACIONADO ACTUALIZADO'] = data.apply(lambda fila: fila['PRODUCTO RELACIONADO'] if frecuencias[fila['RUBRO']] > umbral else 0, axis=1)

X = data['JUSTIFICACION']

data['Etiquetas'] = list(zip(data['RUBRO'], data['PRODUCTO RELACIONADO']))



frecuencias = data['Etiquetas'].value_counts()
# umbral = frecuencias.quantile(0.8)
# print(umbral)

data['Etiquetas'] = data.apply(lambda fila: fila['Etiquetas'] if frecuencias[fila['Etiquetas']] > 90 else (0, 0), axis=1)


# print(data['Etiquetas'].value_counts())
# print(data['Etiquetas'])



# Procesamiento de las etiquetas
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(data['Etiquetas'])

# División del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


 
# # param_grid = {
# #     'clf__n_estimators': [300, 400, 500],
# #     'clf__max_depth': [30, 40, None],
# #     'clf__min_samples_split': [2, 5, 10],
# #     'clf__min_samples_leaf': [1, 2, 4],
# #     'clf__max_features': ['sqrt']  
# # }

param_grid = {
    'n_estimators': 400,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt' 
}


# grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
# grid_search.fit(X_train, y_train)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(random_state=42, **param_grid))
])


# print("Mejores parámetros: ", grid_search.best_params_)


# predictions = grid_search.best_estimator_.predict(X_test)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

for i, class_label in enumerate(mlb.classes_):
    print(f"Clasificación para la etiqueta {class_label}:")
    print(classification_report(y_test[:, i], predictions[:, i]))


nueva_justificacion = """COSMOLAC S.A.S Somos una empresa constituida desde el año 2013 dedicados a la elaboración y distribución de productos lácteos en polvo de excelente calidad garantizando la inocuidad y la satisfacción de los clientes, nuestros productos son: leche entera fortificada con H y V Fortificada con vitaminas (A y D3) y hierro aminoquelado, lo que incrementa el valor nutricional. Leche entera azucarada mezclada con azúcar pulverizada en una proporción del (1%). Alimento lácteo Producto obtenido a partir de la mezcla balanceada de leches en polvo, maltodextrina, suero lácteo y grasa. Dicha actividad es desarrollada en el municipio de Zipaquirá, Cundinamarca. Para el 2024, se tiene proyectado un valor de compras al sector agropecuario nacional de $182.365 millones de pesos de leche cruda. """
justificacion_limpia = limpiar_texto(nueva_justificacion)



prediccion = pipeline.predict([justificacion_limpia])

prediccion_rubro = prediccion[0][0]
prediccion_producto = prediccion[0][1]

if prediccion_rubro == 0:
    print("Rubro: Preguntar al analista")
else:
    print(f"Rubro: {prediccion_rubro}")

print(f"Producto relacionado: {prediccion_producto}")
