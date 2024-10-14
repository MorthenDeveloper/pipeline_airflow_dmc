import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Como ejemplo
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
from joblib import dump, load
import numpy as np

MODEL_PATH= "dags/data/dmc_random_forest_classifier_model.joblib"

class MLSystem:
    def __init__(self):
        self.model = None

    def train_kaggle(self, data_path: str):

        #Cargamos la data de test
        test_data_path = os.path.join("dags", "data", "test.csv")
        test_df = pd.read_csv(test_data_path)

        # Cargar los datos
        df = pd.read_csv(data_path)

        # Dividimos los datos entre características y etiquetas (esto dependerá de la estructura del dataset de Kaggle)
        X = df.drop(columns=["loan_status"],axis=1)  # Aquí 'target' es el nombre de la columna de las etiquetas (ajústalo según el dataset)
        y = df['loan_status']
        
        #Para cambiar las columnas categóricas a numéricas
        label_encoder = LabelEncoder()
        cat_col=X.select_dtypes(exclude=np.number).columns
        for col in cat_col:
            X[col]=label_encoder.fit_transform(X[col])
            test_df[col]=label_encoder.fit_transform(test_df[col])

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar un modelo de ejemplo (RandomForest)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluar el modelo en los datos de prueba
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Precisión: {accuracy}')

        return self.model

    def save_model(self, model_path: str):
        # Guardar el modelo entrenado
        if self.model:
            dump(self.model, model_path)
            print(f'Modelo guardado en {model_path}')
        else:
            print("No se encontró ningún modelo para guardar.")

    def load_model(self, model_path: str):
        # Cargar el modelo guardado
        if os.path.exists(model_path):
            self.model = load(model_path)
            print(f'Modelo cargado desde {model_path}')
            return self.model
        else:
            print(f"No se encontró el modelo en {model_path}")
            return None