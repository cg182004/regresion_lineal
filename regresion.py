import pandas as pd
import os
import random

# Cargar datos
df = pd.read_csv('trat.csv')

# Función para obtener un tratamiento y dieta recomendada basado en el tipo de cirugía
def get_recommendation(tipo_cx):
    # Filtrar el DataFrame para el tipo de cirugía especificado
    filtered_df = df[df['Tipo_cx'] == tipo_cx]
    
    # Verificar si hay resultados para el tipo de cirugía
    if len(filtered_df) == 0:
        print("No se encontraron recomendaciones para el tipo de cirugía especificado.")
        return
    
    # Seleccionar aleatoriamente uno de los tratamientos y dietas disponibles
    random_choice = random.choice(range(len(filtered_df)))
    tratamiento_recomendado = filtered_df.iloc[random_choice]['Tratamiento']
    dieta_recomendada = filtered_df.iloc[random_choice]['Dieta_recomendada']
    
    # Imprimir la recomendación
    print(f"Para el tipo de cirugía {tipo_cx}, se recomienda el siguiente tratamiento: {tratamiento_recomendado}")
    print(f"Dieta recomendada: {dieta_recomendada}")

# Solicitar al usuario que ingrese el tipo de cirugía
tipo_cx_usuario = input("Ingrese el tipo de cirugía: ")
get_recommendation(tipo_cx_usuario)
