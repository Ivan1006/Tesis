import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import re
from openpyxl import load_workbook
from PyPDF2 import PdfReader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
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
    data['Etiquetas'] = data['Etiquetas'].apply(lambda x: [str(tup) for tup in x])
    y = mlb.fit_transform(data['Etiquetas'])
    print(data['Etiquetas'].value_counts())
    # División del conjunto de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    


    # param_grid = {
    #     'clf__n_estimators': [300, 400, 500],
    #     'clf__max_depth': [30, 40, None],
    #     'clf__min_samples_split': [2, 5, 10],
    #     'clf__min_samples_leaf': [1, 2, 4],
    #     'clf__max_features': ['sqrt']  
    # }

    # pipeline = Pipeline([
    #     ('tfidf', TfidfVectorizer(max_df=0.8,min_df=0.01)),
    #     ('clf', RandomForestClassifier(random_state=42))
    # ])


    mejores_parametros_rf = {
        'n_estimators': 400,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt' 
    }

    # grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=4, verbose=2)
    # grid_search.fit(X_train, y_train)
    # print("Mejores parámetros: ", grid_search.best_params_)

    pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.8,min_df=0.01)),
    ('clf', RandomForestClassifier(random_state=42, **mejores_parametros_rf))
])

    pipeline.fit(X_train, y_train)
    predictions= pipeline.predict(X_test)
    probabilities_rf = pipeline.predict_proba(X_test)

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
        classes = np.arange(len(mlb.classes_))
        df_rf =(classification_report(y_test, predictions,output_dict=True))
        df_rf = pd.DataFrame(df_rf).transpose()
        df_rf.to_excel(r"exceles_info/classification_report_model1.xlsx")
    
        # Calculando y visualizando la matriz de confusión para cada clase
    for i, class_label in enumerate(mlb.classes_):
        cm = confusion_matrix(y_test[:, i], predictions[:, i])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión para la etiqueta: {class_label}')
        plt.ylabel('Verdaderos')
        plt.xlabel('Predicciones')
        plt.show()




    # Para Random Forest ROC

    classes = np.arange(len(mlb.classes_))
    y_test_binarized = label_binarize(y_test, classes=classes)


    n_classes = len(probabilities_rf) 
    colors = cycle(['blue', 'red', 'green', 'purple', 'orange'])
    plt.figure(figsize=(7, 5))

    for i, color in zip(range(n_classes), colors):
        probas_ = probabilities_rf[i][:, 1]
        fpr, tpr, thresholds = roc_curve(y_test_binarized[:, i], probas_)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve - Random Forest')
    plt.legend(loc="lower right")
    plt.show()
    return y_test, probabilities_rf, mlb.classes_



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
