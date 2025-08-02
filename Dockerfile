# Author: Vikash Kumar
# Purpose: Container to run predict.py inside an isolated environment

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ /app/src/
COPY src/model.joblib /app/src/model.joblib
COPY src/quant_params.joblib /app/src/quant_params.joblib

CMD ["python", "src/predict.py"]

