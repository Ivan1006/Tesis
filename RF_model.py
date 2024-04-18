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
import numpy as np

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

def RF_model(data):
    #Procesamiento de las etiquetas
    X = data['JUSTIFICACION']
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(data['Etiquetas'])

    print(data['Etiquetas'].value_counts())
    # División del conjunto de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    param_grid = {
        'clf__n_estimators': [300, 400, 500],
        'clf__max_depth': [30, 40, None],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__max_features': ['sqrt']  
    }

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_df=0.8,min_df=0.01)),
        ('clf', RandomForestClassifier(random_state=42))
    ])


    # param_grid = {
    #     'n_estimators': 400,
    #     'max_depth': None,
    #     'min_samples_split': 2,
    #     'min_samples_leaf': 1,
    #     'max_features': 'sqrt' 
    # }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=4, verbose=2)
    grid_search.fit(X_train, y_train)
    print("Mejores parámetros: ", grid_search.best_params_)


    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    dfs = []

    for i, class_label in enumerate(mlb.classes_):
        print(f"Clasificación para la etiqueta {class_label}:")
        report_dict = classification_report(y_test[:, i], predictions[:, i], output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict)
        report_df = report_df.transpose()
        report_df['Etiqueta'] = class_label
        dfs.append(report_df)

    # Concatena todos los DataFrames en uno solo
    final_report_df = pd.concat(dfs)

    # Guarda el DataFrame final en un archivo Excel
    final_report_df.to_excel(r"exceles_info/classification_reports.xlsx")


# nueva_justificacion = """COSMOLAC S.A.S Somos una empresa constituida desde el año 2013 dedicados a la elaboración y distribución de productos lácteos en polvo de excelente calidad garantizando la inocuidad y la satisfacción de los clientes, nuestros productos son: leche entera fortificada con H y V Fortificada con vitaminas (A y D3) y hierro aminoquelado, lo que incrementa el valor nutricional. Leche entera azucarada mezclada con azúcar pulverizada en una proporción del (1%). Alimento lácteo Producto obtenido a partir de la mezcla balanceada de leches en polvo, maltodextrina, suero lácteo y grasa. Dicha actividad es desarrollada en el municipio de Zipaquirá, Cundinamarca. Para el 2024, se tiene proyectado un valor de compras al sector agropecuario nacional de $182.365 millones de pesos de leche cruda. """
# justificacion_limpia = limpiar_texto(nueva_justificacion)



# prediccion = pipeline.predict([justificacion_limpia])

# prediccion_rubro = prediccion[0][0]
# prediccion_producto = prediccion[0][1]

# if prediccion_rubro == 0:
#     print("Rubro: Preguntar al analista")
# else:
#     print(f"Rubro: {prediccion_rubro}")

# print(f"Producto relacionado: {prediccion_producto}")
