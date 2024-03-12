from PyPDF2 import PdfReader
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import re
import fitz

# Configuración de PyTorch para usar GPU si está disponible
dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'

# Modelo y Tokenizer de BERT
modelo_nombre = 'bert-large-uncased-whole-word-masking-finetuned-squad'
tokenizer = BertTokenizer.from_pretrained(modelo_nombre)
modelo = BertForQuestionAnswering.from_pretrained(modelo_nombre).to(dispositivo)

def extraer_y_guardar_texto_pdf(ruta_pdf, ruta_txt):
    documento = fitz.open(ruta_pdf)
    texto_acumulado = ""
    
    for pagina in documento:
        texto_pagina = pagina.get_text()
        # Limpieza más exhaustiva: eliminar espacios en blanco extra al inicio y al final de cada línea,
        # y reemplazar múltiples espacios en blanco seguidos por uno solo.
        texto_limpio = re.sub(r'\s+', ' ', texto_pagina).strip()
        texto_acumulado += texto_limpio + "\n--- Fin de Página ---\n"
    
    # Asegúrate de que el archivo se sobrescribe si ya existe
    with open(ruta_txt, "w", encoding="utf-8") as archivo_salida:
        archivo_salida.write(texto_acumulado)
    
    documento.close()
    return texto_acumulado

def dividir_en_segmentos(texto, longitud_max=512, longitud_overlap=50):
    palabras = texto.split(' ')
    segmentos = []
    segmento_actual = ''
    ultimo_corte = 0

    for i, palabra in enumerate(palabras):
        if len(segmento_actual) + len(palabra) < longitud_max or i - ultimo_corte < longitud_overlap:
            segmento_actual += palabra + ' '
        else:
            segmentos.append(segmento_actual)
            ultimo_corte = i
            segmento_actual = palabra + ' '
    if segmento_actual:
        segmentos.append(segmento_actual)
    return segmentos


def responder_pregunta(pregunta, contexto):
    inputs = tokenizer.encode_plus(pregunta, contexto, add_special_tokens=True, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(dispositivo) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = modelo(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    
    # Verifica si se encontró una respuesta válida
    if answer_start < answer_end:
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
        return answer.strip()
    else:
        return "Respuesta no encontrada"

# Ejemplo simplificado para probar una pregunta y contexto específicos
pregunta_ejemplo = "¿Cuál es el valor del Coste de Ventas del último año fiscal reportado?"
contexto_ejemplo = "El Coste de Ventas del último año fiscal reportado fue de $120,000."
respuesta_ejemplo = responder_pregunta(pregunta_ejemplo, contexto_ejemplo)
print(respuesta_ejemplo)





def responder_preguntas(segmentos):
    preguntas = {
        "Coste de Ventas": "¿Cuál es el valor del Coste de Ventas del último año fiscal reportado?",
        "Gastos Administración": "¿Cuánto ascienden los Gastos de Administración en el más reciente estado financiero anual?",
        "Gastos de Ventas": "¿Cuál fue el total de Gastos de Ventas durante el último período contable cerrado?",
        "Activos": "¿Cuál es el monto total de Activos registrados en el balance más reciente?",
        "Ingresos": "¿Cuál es el monto de Ingresos Brutos Anuales según el último informe financiero?",

    }

    respuestas = {}
    respuestas_encontradas = set()

    for segmento in segmentos:
        for clave, pregunta in preguntas.items():
            if clave not in respuestas_encontradas:
                respuesta = responder_pregunta(pregunta, segmento)
                if respuesta and respuesta != "No se encontró un número":
                    respuestas[clave] = respuesta
                    respuestas_encontradas.add(clave)

    return respuestas

# Ruta del archivo PDF
ruta_pdf = r'D:\Tesis maestria\Langchain\pdfs\Estados Financieros Inveragro la Acacia S.A.S. a Jun 2019.pdf'
ruta_txt = r"D:\Tesis maestria\Langchain\pdfs/output.txt"  # Actualiza esto con la ruta deseada para el archivo de salida

# Extraer y limpiar el texto del PDF
texto_pdf = extraer_y_guardar_texto_pdf(ruta_pdf, ruta_txt)

# Dividir el texto en segmentos
segmentos_texto = dividir_en_segmentos(texto_pdf)

# Obtener respuestas a las preguntas
informacion = responder_preguntas(segmentos_texto)
# Imprimir las respuestas
for clave, respuesta in informacion.items():
    print(f"{clave}: {respuesta}")
