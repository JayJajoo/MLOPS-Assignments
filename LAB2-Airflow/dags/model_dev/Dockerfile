FROM apache/airflow:2.9.1-python3.11

# Install extra Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy DAGs and training scripts
WORKDIR /app
COPY . /app