import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import os

# Cargar datos
df = pd.read_csv('trat.csv')

# Comprobar que el DataFrame tiene las columnas necesarias
if 'Tipo_cx' not in df.columns or 'Tratamiento' not in df.columns or 'Dieta_recomendada' not in df.columns:
    print("El DataFrame no contiene las columnas necesarias para realizar predicciones.")
    exit()

# Preprocesamiento de datos
# Codificar la columna 'Tipo_cx' que es categórica
encoder = OneHotEncoder(sparse_output=False)  # Usar sparse_output=False para evitar futuros warnings
encoded_features = encoder.fit_transform(df[['Tipo_cx']])

# Crear un DataFrame con las características codificadas
X = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Tipo_cx']))
y_treatment = df['Tratamiento']  # Establecer el objetivo de predicción para el tratamiento
y_diet = df['Dieta_recomendada']  # Establecer el objetivo de predicción para la dieta

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_treatment_train, y_treatment_test, y_diet_train, y_diet_test = train_test_split(
    X, y_treatment, y_diet, test_size=0.2, random_state=42)

# Entrenar un modelo de regresión logística para el tratamiento
model_treatment = LogisticRegression(max_iter=1000)
model_treatment.fit(X_train, y_treatment_train)

# Entrenar un modelo de regresión logística para la dieta
model_diet = LogisticRegression(max_iter=1000)
model_diet.fit(X_train, y_diet_train)

# Función para realizar predicciones basadas en el tipo de cirugía
def predict_treatment_and_diet(tipo_cx):
    # Crear DataFrame con el mismo formato de columnas que se usó para entrenar el modelo
    input_data = pd.DataFrame([[tipo_cx]], columns=['Tipo_cx'])
    input_encoded = encoder.transform(input_data)
    input_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(['Tipo_cx']))

    # Predecir el tratamiento y la dieta
    predicted_treatment = model_treatment.predict(input_df)
    predicted_diet = model_diet.predict(input_df)
    
    print(f"\n\nPara el tipo de cirugía '{tipo_cx}':")
    print(f"Tratamiento recomendado: {predicted_treatment[0]}")
    print(f"Dieta recomendada: {predicted_diet[0]}")

# Solicitar al usuario que ingrese el tipo de cirugía
tipo_cx_usuario = input("Ingrese el tipo de cirugía: ")
predict_treatment_and_diet(tipo_cx_usuario)
