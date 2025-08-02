# 📘 MLOps Major Assignment – PG Diploma in Data Engineering

This repository contains my solution for the MLOps major exam assignment. It demonstrates an end-to-end ML workflow using Linear Regression, including training, quantization, Docker setup, CI pipeline, and testing.

---

## 🔧 Project Overview

- 🔁 Model training and evaluation using `scikit-learn`
- 🧪 Model quantization by manually converting coefficients to `uint8`
- 📦 Docker containerization for easy deployment
- ⚙️ GitHub Actions CI/CD pipeline for automated testing and reproducibility

---

## 📁 Folder Structure

# MLOps Major Assignment\nThis repository contains the end-to-end MLOps workflow for training, testing, quantizing, and deploying a linear regression model.
├── .github/
│ └── workflows/
│ └── mlops.yml # CI workflow
├── src/
│ ├── train.py # Training script
│ ├── test_train.py # Unit test for training
│ ├── predict.py # Prediction using trained model
│ ├── quantize.py # Manual quantization
│ ├── model.joblib # Trained model
│ ├── quant_params.joblib # Quantized weights
│ └── unquant_params.joblib # Original weights
├── Dockerfile # Docker setup
├── requirements.txt # Python dependencies
└── README.md # This file

🧠 Learnings
This assignment gave hands-on practice with:

Managing multiple branches for MLOps lifecycle stages
Working with .gitignore, GitHub Actions, and Python testing
Manual quantization to optimize model storage
End-to-end reproducible ML pipeline using Git and Docker

🧑‍💻 Author
Name: Vikash Kumar
Program: PG Diploma in Data Engineering
Institution: IIT Jodhpur
Assignment: MLOps – Major Exam Submission (Trimester 3)

