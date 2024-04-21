import tensorflow as tf
import pandas as pd
import numpy as np

# Paso 1: Carga y Preparación de los Datos
# df = pd.read_csv('tratamiento.csv')

# otra forma de ubicar el archivo tratamiento.csv
df = pd.read_csv(r"C:\\Users\\luisa\\OneDrive\\Documentos\\GitHub\\regresion_lineal\\tratamiento.csv")

# Asumiendo que tienes una columna para predecir que podría ser algo como 'ResultadoTratamiento'
datos_entrenamiento = df[['Tipo_cx','Tratamiento','Dieta_recomendada','ResultadoTratamiento']].values
etiquetas_entrenamiento = df['ResultadoTratamiento'].values


# Convertimos los datos a tipos flotantes
datos_entrenamiento = datos_entrenamiento.astype(float)
etiquetas_entrenamiento = etiquetas_entrenamiento.astype(float)

# Normalización de los datos
media = np.mean(datos_entrenamiento, axis=0)
desviacion_std = np.std(datos_entrenamiento, axis=0)

datos_entrenamiento = (datos_entrenamiento - media) / desviacion_std

# Paso 2: Creación del Modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(3,))
])

# Configuración del modelo
modelo.compile(optimizer='adam', loss='mae', metrics=['mae'])

# Paso 3: Entrenamiento del Modelo
modelo.fit(datos_entrenamiento, etiquetas_entrenamiento, epochs=100)

# Paso 4: Evaluación del Modelo
# Aquí deberías dividir tus datos en entrenamiento y prueba para evaluar el modelo adecuadamente
loss, mae = modelo.evaluate(datos_entrenamiento, etiquetas_entrenamiento)
print("Pérdida del modelo:", loss)
print("Error absoluto medio:", mae)