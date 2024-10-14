import os
from kaggle.api.kaggle_api_extended import KaggleApi
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
from joblib import dump, load
import numpy as np

MODEL_PATH= "dags/data/dmc_random_forest_classifier_model.joblib"

#Variables de entorno de credenciales de Kaggle
os.environ['KAGGLE_USERNAME'] = 'fabian426'
os.environ['KAGGLE_KEY'] = 'e7eb2c70bc652c11cb51980cce0b6d8c'

competition = 'playground-series-s4e10'

# Rutas
MODEL_PATH = os.path.join('opt','airflow','dags', 'data', 'model_dmc.joblib')
TRAIN_DATA_PATH = os.path.join('opt','airflow','dags', 'data', 'train.csv')
TEST_DATA_PATH = os.path.join('opt','airflow','dags', 'data', 'test.csv')

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


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 14),
    'email ':['fabianvillanuevajaqui@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'kaggle_ml_pipeline',
    default_args=default_args,
    description='A simple ML pipeline',
    schedule_interval='0 * * * *'
)

def GetDataKaggle(**kwargs):

    # Inicia la API de Kaggle
    api = KaggleApi()
    api.authenticate()

    # Define la ruta donde se guardarán los archivos, relativa a la raíz del proyecto
    data_path = os.path.join("dags", "data")

    # Crea la ruta si no existe
    os.makedirs(data_path, exist_ok=True)

    # Descargar los archivos train.csv y test.csv a la ruta especificada
    competition_name = "playground-series-s4e10"
    api.competition_download_file(competition_name, "train.csv", path=data_path)
    api.competition_download_file(competition_name, "test.csv", path=data_path)

    # Ruta completa al archivo train.csv
    train_file_path = os.path.join(data_path, "train.csv")

    return train_file_path

# Función para entrenar el modelo usando la clase MLSystem
def ml_kaggle(**kwargs):
    ml_system = MLSystem()
    data_path = kwargs['ti'].xcom_pull(task_ids='GetDataKaggle')  # Obtenemos la ruta de train.csv desde el XCom
    model = ml_system.train_kaggle(data_path=data_path)
    kwargs['ti'].xcom_push(key='model', value=model)

# Función para guardar el modelo
def submit_model(**kwargs):
    ml_system = MLSystem()
    model = kwargs['ti'].xcom_pull(key='model')
    if model:
        ml_system.model = model
        ml_system.save_model(MODEL_PATH)
    else:
        print("No model found to save.")

# Nueva función para evaluar el modelo con el archivo test.csv
def evaluate_model(**kwargs):
    model_path = MODEL_PATH
    test_data_path = kwargs['ti'].xcom_pull(task_ids='GetDataKaggle')

    # Cargar el modelo entrenado
    ml_system = MLSystem()
    ml_system.model = ml_system.load_model(model_path)

    # Cargar los datos de prueba
    df_test = pd.read_csv(test_data_path)

    # Asumimos que test.csv tiene las mismas características que train.csv menos la columna 'target'
    X_test = df_test.drop('target', axis=1)
    y_test = df_test['target']

    # Hacer predicciones
    y_pred = ml_system.model.predict(X_test)

    # Calcular la precisión
    accuracy = accuracy_score(y_test, y_pred)
    #print(f'Evaluation accuracy: {accuracy}')

get_data = PythonOperator(
    task_id='GetDataKaggle',
    python_callable=GetDataKaggle,
    provide_context=True,
    dag=dag
)
train_model = PythonOperator(
    task_id='ML_Kaggle',
    python_callable=ml_kaggle,
    provide_context=True ,
    dag=dag
)
submit_model = PythonOperator(
    task_id='SubmitModel',
    python_callable=submit_model,
    provide_context=True,
    dag=dag
)
evaluate_model = PythonOperator(
    task_id='EvaluateModel',
    python_callable=evaluate_model,
    provide_context=True,
    dag=dag
)
# Definir la secuencia de ejecución
get_data >> train_model >> submit_model >> evaluate_model