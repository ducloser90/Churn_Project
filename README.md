# Telco Customer Churn Prediction

A machine-learning project for predicting whether a telco customer is likely to churn. The repository combines a data validation and preprocessing pipeline, model training with XGBoost, MLflow experiment tracking, a FastAPI inference service, a Gradio interface for interactive prediction, and GitHub Actions automation for Docker image builds and publishing.

## Overview

The project takes a Telco Customer Churn dataset, validates the data quality, transforms the raw customer records into model-ready features, trains an XGBoost classifier, and exposes predictions through a web API and a browser-based UI. It also includes a GitHub Actions workflow that builds and publishes the Docker image to Docker Hub whenever changes are pushed to the main branch.

The implementation uses:

- Python and pandas for data processing
- Great Expectations for dataset validation
- scikit-learn for evaluation and train/test splitting
- XGBoost for the classifier
- MLflow for experiment tracking and model storage
- FastAPI for the prediction API
- Gradio for the interactive UI
- Docker for containerization
- GitHub Actions for automated Docker image builds and publishing

## Repository structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── artifacts/
│   └── feature_columns.json
├── data/
│   ├── processed/
│   │   └── telco_churn_processed.csv
│   └── raw/
│       └── data.csv
├── mlruns/
│   └── MLflow experiment artifacts and metadata
├── notebooks/
│   └── eda.ipynb
├── pipelines/
│   └── run_pipeline.py
├── src/
│   ├── app/
│   │   └── main.py
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess_data.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── evaluate.py
│   │   ├── train.py
│   │   └── tune.py
│   ├── serving/
│   │   ├── inference.py
│   │   └── model/
│   │       ├── MLmodel
│   │       ├── conda.yaml
│   │       ├── feature_columns.txt
│   │       ├── model.ubj
│   │       ├── python_env.yaml
│   │       └── requirements.txt
│   └── utils/
│       └── validate_data.py
├── .dockerignore
├── .gitignore
├── dockerfile
├── requirements.txt
├── README.md
└── mlflow.db
```

## Architecture

The project follows a simple end-to-end ML workflow:

1. Data is loaded from the raw CSV file in [src/data/load_data.py](src/data/load_data.py).
2. The dataset is validated with Great Expectations in [src/utils/validate_data.py](src/utils/validate_data.py).
3. Preprocessing and feature engineering are performed in [src/data/preprocess_data.py](src/data/preprocess_data.py) and [src/features/build_features.py](src/features/build_features.py).
4. The end-to-end training pipeline is implemented in [pipelines/run_pipeline.py](pipelines/run_pipeline.py).
5. Model evaluation and optional tuning live in [src/models/evaluate.py](src/models/evaluate.py) and [src/models/tune.py](src/models/tune.py).
6. Production inference is handled in [src/serving/inference.py](src/serving/inference.py), which loads the model from the local model directory or the latest MLflow artifact.
7. The API and Gradio UI are served from [src/app/main.py](src/app/main.py).

## Data pipeline

The raw data is stored in [data/raw/data.csv](data/raw/data.csv). The project preprocessing stage writes a cleaned dataset to [data/processed/telco_churn_processed.csv](data/processed/telco_churn_processed.csv).

### Validation and preprocessing

The validation logic checks required columns, allowed category values, and basic numeric constraints. It verifies things such as:

- customerID presence and non-null values
- valid values for gender, Partner, Dependents, PhoneService, Contract, and InternetService
- tenure, MonthlyCharges, and TotalCharges ranges
- total charges not being less than monthly charges in the majority of records

After validation, preprocessing normalizes column names, drops identifier columns if present, converts the target variable from Yes/No to 0/1, converts TotalCharges to numeric values, fills missing numerics, and preserves a deterministic feature schema for serving.

### Feature engineering

Feature engineering uses a deterministic approach:

- binary categorical columns are encoded to 0/1 values
- more complex categorical columns are one-hot encoded with drop_first=True
- numeric columns are kept in their numeric form
- generated feature metadata is saved to [artifacts/feature_columns.json](artifacts/feature_columns.json)

The feature metadata is also logged to MLflow as a text artifact named feature_columns.txt.

## ML model and training flow

The training entry point is [pipelines/run_pipeline.py](pipelines/run_pipeline.py).

This pipeline:

- loads a CSV via [src/data/load_data.py](src/data/load_data.py)
- validates the dataset
- preprocesses and saves processed data
- builds feature columns
- splits data into train/test sets
- fits an XGBoost classifier
- computes precision, recall, F1 score, and ROC AUC
- logs the run to MLflow
- saves the trained model as an MLflow artifact

The model used is XGBClassifier with a threshold set by the CLI option --threshold. The pipeline logs metrics including:

- data_quality_pass
- train_time
- pred_time
- precision
- recall
- f1
- roc_auc

The configuration stores MLflow artifacts in the local mlruns directory and also uses a local SQLite database file named mlflow.db.

## Model serving and API

The inference service is implemented in [src/serving/inference.py](src/serving/inference.py).

Key behavior:

- It loads a production model from [src/serving/model](src/serving/model)
- If that local model is unavailable, it falls back to the latest MLflow artifact under mlruns
- It transforms incoming raw customer data into the exact schema expected by the trained model
- It reindexes the feature set to match the saved training columns
- It returns a human-readable business result: Likely to churn or Not likely to churn

### API app

The FastAPI app is defined in [src/app/main.py](src/app/main.py).

Endpoints:

- GET /
  - Returns service health status as JSON: {"status": "ok"}
- POST /predict
  - Accepts a JSON body that follows the CustomerData schema
  - Returns a prediction payload with a prediction field

The request schema includes fields such as:

- gender
- Partner
- Dependents
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod
- tenure
- MonthlyCharges
- TotalCharges

### Gradio interface

The same app also mounts a Gradio interface at /ui. This UI lets a user fill in customer information through a form and receive a churn prediction in the browser.

## Environment and dependencies

Project dependencies are managed in [requirements.txt](requirements.txt). The stack includes:

- fastapi
- uvicorn
- gradio
- pandas
- scikit-learn
- xgboost
- great-expectations
- mlflow
- optuna
- Flask, flask-cors
- joblib

This repository is built around a local Python environment and does not define a pyproject.toml or poetry configuration.

## Local setup

From the repository root, install dependencies:

```bash
pip install -r requirements.txt
```

A virtual environment is used in the current workspace and was activated with:

```bash
source .venv/bin/activate
```

## Training

The training pipeline is launched from [pipelines/run_pipeline.py](pipelines/run_pipeline.py).

Example run:

```bash
python pipelines/run_pipeline.py --input data/raw/data.csv --target Churn --threshold 0.35 --test_size 0.2 --experiment "Telco Churn"
```

Supported CLI arguments:

- --input: path to the CSV file
- --target: target column name (default: Churn)
- --threshold: classification threshold (default: 0.35)
- --test_size: train/test split proportion (default: 0.2)
- --experiment: MLflow experiment name (default: Telco Churn)
- --mlflow_uri: optional MLflow tracking URI; defaults to the local mlruns directory

## Running the API locally

The Dockerfile starts the application with:

```bash
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

This is the runtime command used by the container image. Once the app is running, the service is available on the default FastAPI host and port:

- http://localhost:8000/
- http://localhost:8000/predict
- http://localhost:8000/ui

## Example prediction request

Example JSON payload for POST /predict:

```json
{
  "gender": "Female",
  "Partner": "No",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "tenure": 1,
  "MonthlyCharges": 85.0,
  "TotalCharges": 85.0
}
```

Example response:

```json
{
  "prediction": "Likely to churn"
}
```

## Docker configuration

The repository includes a Dockerfile at [dockerfile](dockerfile).

The image:

- uses Python 3.13 slim
- installs the packages from [requirements.txt](requirements.txt)
- sets PYTHONPATH to /app
- exposes port 8000
- copies the model artifacts from [src/serving/model](src/serving/model) into /app/model before launch
- starts the app with uvicorn against src.app.main:app

## GitHub Actions workflow

The CI workflow is defined in [.github/workflows/ci.yml](.github/workflows/ci.yml).

It is triggered on pushes to the main branch and performs the following actions:

1. Checks out the repository
2. Sets up Docker Buildx
3. Logs into Docker Hub using repository secrets named DOCKERHUB_USERNAME and DOCKERHUB_TOKEN
4. Builds and pushes the Docker image to Docker Hub with the tag duclo90/churn-api:latest


## Notes on MLflow and artifacts

The project writes run metadata to the local MLflow directory under [mlruns](mlruns). It also stores the trained model and schema artifacts in [src/serving/model](src/serving/model), including:

- MLmodel
- model.ubj
- feature_columns.txt
- conda.yaml
- python_env.yaml
- requirements.txt

These artifacts are used for consistent feature alignment during inference.


