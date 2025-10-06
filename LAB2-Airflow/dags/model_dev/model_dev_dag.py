from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from model_dev import train  

# DAG definition
with DAG(
    dag_id="model_dev_dag",
    start_date=datetime(2025, 10, 6),
    schedule=None,  # manual trigger
    catchup=False,
    tags=["ml", "training"],
) as dag:

    # Task: Load raw data
    load_data_task = PythonOperator(
        task_id="load_data",
        python_callable=train.load_data,
    )

    # Task: Split dataset
    split_data_task = PythonOperator(
        task_id="split_dataset",
        python_callable=train.split_dataset,
    )

    # Task: Train model and save model path to XCom
    def train_model_callable(**kwargs):
        model_path = train.train_model()
        # Push the model path to XCom
        kwargs['ti'].xcom_push(key='model_path', value=model_path)

    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model_callable
    )

    # Task: Evaluate model
    def evaluate_model_callable(**kwargs):
        ti = kwargs['ti']
        model_path = ti.xcom_pull(task_ids='train_model', key='model_path')
        report = train.evaluate_model(model_path)
        # Push evaluation report to XCom
        ti.xcom_push(key='eval_report', value=report)

    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model_callable
    )

    # Task: Send evaluation report via email
    def send_email_callable(**kwargs):
        ti = kwargs['ti']
        report = ti.xcom_pull(task_ids='evaluate_model', key='eval_report')
        train.send_email(
            report,
            sender_email="jayjajoo02@gmail.com",
            receiver_email="jayjajoousa02@gmail.com",
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            login="jayjajoo02@gmail.com",
            password="YOUR_APP_PASSWORD"  # Gmail app password
        )

    email_task = PythonOperator(
        task_id="send_email",
        python_callable=send_email_callable
    )

    # DAG task dependencies
    load_data_task >> split_data_task >> train_model_task >> evaluate_task >> email_task
