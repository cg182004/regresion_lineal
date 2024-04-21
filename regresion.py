import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# Cambiar al directorio correcto
os.chdir(r'C:\Users\cleve\Documents\GitHub\regresion_lineal')

# Cargar datos
df = pd.read_csv('trat.csv')
print(df.head())
print(df.describe())

# Configurar OneHotEncoder
ohe = OneHotEncoder(sparse=False)

# Codificar variables categóricas
features_to_encode = ['Tipo_cx', 'Tratamiento', 'Dieta_recomendada']
encoded_features = ohe.fit_transform(df[features_to_encode])

