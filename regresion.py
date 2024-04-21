import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import os

# Cambiar al directorio correcto
os.chdir(r'C:\Users\cleve\Documents\GitHub\regresion_lineal')

# Cargar datos
df = pd.read_csv('trat.csv')
print(df.head())
print(df.describe())

# Configurar OneHotEncoder
ohe = OneHotEncoder(sparse=False)

# Variables categóricas para codificar
features_to_encode = ['Tipo_cx', 'Dieta_recomendada']
df_encoded = pd.DataFrame(ohe.fit_transform(df[features_to_encode]))
df_encoded.columns = ohe.get_feature_names_out(features_to_encode)  # Actualizado para usar get_feature_names_out

# Concatenar las características codificadas con las demás no categóricas (si las hay)
df_final = pd.concat([df_encoded, df[['Tratamiento']]], axis=1)  # Mantenemos 'Tratamiento' para entrenamiento

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df_final.drop(columns=['Tratamiento']), df['Tratamiento'], test_size=0.2, random_state=42)

# Crear y entrenar el modelo de clasificación
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predecir los tratamientos en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
