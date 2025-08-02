# Purpose: Container to run predict.py inside an isolated environment

# Author: Vikash Kumar
# Purpose: Container to run predict.py inside an isolated environment

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ /app/src/

CMD ["python", "src/predict.py"]

