import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_experiment("breast-cancer-mlops")

with mlflow.start_run():
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 2000)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("roc_auc", auc)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.joblib")
    mlflow.sklearn.log_model(model, artifact_path="model")

    print("✅ Training done")
    print("accuracy:", acc, "f1:", f1, "roc_auc:", auc)
    print("Saved model to models/model.joblib")

