# Dockerfile
# Author: Vikash Kumar
# Purpose: Container to run predict.py inside an isolated environment

FROM python:3.10

WORKDIR /app

# Copy all source files
COPY src/ ./src/

# Set working directory inside container
WORKDIR /app/src

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Run prediction script
CMD ["python", "predict.py"]

