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
from data.utils.ml import MLSystem

MODEL_PATH= "dags/data/dmc_random_forest_classifier_model.joblib"

#Variables de entorno de credenciales de Kaggle
os.environ['KAGGLE_USERNAME'] = 'fabian426'
os.environ['KAGGLE_KEY'] = 'e7eb2c70bc652c11cb51980cce0b6d8c'

competition = 'playground-series-s4e10'

# Rutas
MODEL_PATH = os.path.join('opt','airflow','dags', 'data', 'model_dmc.joblib')
TRAIN_DATA_PATH = os.path.join('opt','airflow','dags', 'data', 'train.csv')
TEST_DATA_PATH = os.path.join('opt','airflow','dags', 'data', 'test.csv')


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

    data_path=TRAIN_DATA_PATH
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
    
    model_file_path = MODEL_PATH
    dump(model, model_file_path)

    kwargs['ti'].xcom_push(key='model', value=model_file_path)

# Función para guardar el modelo
def submit_model(**kwargs):
    ml_system = MLSystem()
    model_file_path = kwargs['ti'].xcom_pull(key='model')
    if model_file_path:
        ml_system.model = load(model_file_path)
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

    X_test = df_test.drop(columns=["loan_status"], axis=1)
    y_test = df_test['loan_status']

    # Hacer predicciones
    y_pred = ml_system.model.predict(X_test)

    # Calcular la precisión
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Evaluation accuracy: {accuracy}')

    return accuracy

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