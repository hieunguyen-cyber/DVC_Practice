import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pickle
import json
import yaml
import pandas as pd
import dagshub
from mlflow.models.signature import infer_signature
import optuna
import os

train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

X_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]
X_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]

dagshub.init(repo_owner='hieunguyen-cyber', repo_name='DVC-MLflow-DAGsHub_Practice', mlflow=True)

mlflow.set_experiment("RFC")

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Convert test data to DataFrame for logging
X_test_df = pd.DataFrame(X_test, columns=X_train.columns)  # Fix lỗi feature names
y_test_df = pd.DataFrame(y_test, columns=["target"])

# Infer model signature
signature = infer_signature(X_test, y_pred)

# Define objective function for hyperparameter tuning
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 2, 20)
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return accuracy_score(y_test, y_pred)

# Optimize hyperparameters using Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# Train the best model with optimal parameters
best_params = study.best_params
model = RandomForestClassifier(**best_params, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# MLflow logging
with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", acc)
    signature = infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(model, "random_forest_model", signature=signature)

    # Save and log test data
    X_test_df.to_csv("X_test.csv", index=False)
    y_test_df.to_csv("y_test.csv", index=False)
    mlflow.log_artifact("X_test.csv")
    mlflow.log_artifact("y_test.csv")

# Save model using DVC
os.makedirs("./model", exist_ok=True)
with open("./model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save metrics
with open("./metrics.json", "w") as f:
    json.dump({"accuracy": acc}, f)

# Clean up temporary files
os.remove("X_test.csv")
os.remove("y_test.csv")
