import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import os

# Cambiar al directorio correcto
os.chdir(r'C:\Users\cleve\Documents\GitHub\regresion_lineal')

# Cargar datos
df = pd.read_csv('trat.csv')

# Comprobar que el DataFrame tiene las columnas necesarias
if 'Tipo_cx' not in df.columns or 'Tratamiento' not in df.columns or 'Dieta_recomendada' not in df.columns:
    print("El DataFrame no contiene las columnas necesarias para realizar predicciones.")
    exit()

# Preprocesamiento de datos
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(df[['Tipo_cx']])
X = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Tipo_cx']))
y_treatment = df['Tratamiento']
y_diet = df['Dieta_recomendada']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_treatment_train, y_treatment_test, y_diet_train, y_diet_test = train_test_split(
    X, y_treatment, y_diet, test_size=0.2, random_state=42)

model_treatment = LogisticRegression(max_iter=1000)
model_treatment.fit(X_train, y_treatment_train)

model_diet = LogisticRegression(max_iter=1000)
model_diet.fit(X_train, y_diet_train)

def predict_treatment_and_diet(tipo_cx):
    input_data = pd.DataFrame([[tipo_cx]], columns=['Tipo_cx'])
    input_encoded = encoder.transform(input_data)
    input_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(['Tipo_cx']))

    predicted_treatment = model_treatment.predict(input_df)
    predicted_diet = model_diet.predict(input_df)
    
    print(f"\nPara el tipo de cirugía '{tipo_cx}':")
    print(f"Tratamiento recomendado: {predicted_treatment[0]}")
    print(f"Dieta recomendada: {predicted_diet[0]}")

def menu():
    print("\nSeleccione el tipo de cirugía:")
    cirugias = {
        '1': 'Reducción de senos',
        '2': 'Levantamiento',
        '3': 'Abdominoplastia',
        '4': 'Liposucción',
        '5': 'Lipoescultura',
        '6': 'Lipo vaser abdominal'
    }
    tipos_paciente = {
        '0': 'normal',
        '1': 'asmático',
        '2': 'hipertenso',
        '3': 'renal',
        '4': 'diabético'  # Nueva opción añadida
    }

    for key, value in cirugias.items():
        print(f"{key}. {value}")

    cirugia_choice = input("\nIngrese el número de su elección de cirugía: ")
    if cirugia_choice in cirugias:
        print("\nSeleccione el tipo de paciente:")
        for key, value in tipos_paciente.items():
            print(f"{key}. {value}")
        
        tipo_paciente_choice = input("\nIngrese el número de su elección de tipo de paciente: ")
        if tipo_paciente_choice in tipos_paciente:
            # Ajusta el formato de 'cirugia_choice' para asegurar que tenga dos dígitos
            codigo_cirugia = f"cx0{cirugia_choice}{tipo_paciente_choice}"
            predict_treatment_and_diet(codigo_cirugia)
        else:
            print("Elección de tipo de paciente no válida, por favor intente de nuevo.")
    else:
        print("Elección de cirugía no válida, por favor intente de nuevo.")

menu()
