import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
df = pd.read_excel(r'Informacion.xlsx')
df = df.dropna()

# Contando la frecuencia de cada producto relacionado
producto_counts = df['PRODUCTO RELACIONADO'].value_counts()

# Crear un DataFrame que asocie cada 'PRODUCTO RELACIONADO' con su 'DESCRIPCION'
descripcion_producto = df.drop_duplicates(subset=['PRODUCTO RELACIONADO']).set_index('PRODUCTO RELACIONADO')['DESCRIPCION']

# Crear una nueva columna para clasificar los productos
df['Clasificación Producto'] = df['PRODUCTO RELACIONADO'].apply(lambda x: "Otros" if producto_counts[x] < 100 else x)

# Asociar la descripción correcta con cada clasificación de producto
df['Descripción Clasificación'] = df['Clasificación Producto'].map(lambda x: descripcion_producto.get(x, "Otros"))

# Recalcular las frecuencias con la nueva clasificación y descripciones
producto_counts_actualizado = df['Descripción Clasificación'].value_counts()

# Gráfico de barras para todos los productos con la nueva clasificación y descripciones
plt.figure(figsize=(15, 10))
sns.barplot(x=producto_counts_actualizado.index, y=producto_counts_actualizado.values, palette="viridis")
plt.title('Frecuencia de Productos con Grupo "Otros"')
plt.xlabel('Descripción del Producto o Grupo "Otros"')
plt.ylabel('Frecuencia')
plt.xticks(rotation=55)
plt.show()
