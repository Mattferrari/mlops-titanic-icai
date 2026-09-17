import sys

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Hiperparametros por linea de comandos: python train.py <n_estimators> <max_depth>
# max_depth = 0 significa sin limite de profundidad
n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
max_depth_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 0
max_depth = max_depth_arg if max_depth_arg > 0 else None

# Cargar y preparar los datos
COLS = ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
df = pd.read_csv("data/titanic.csv")[COLS].dropna()

y = df["survived"]
X = pd.get_dummies(
    df.drop(columns=["survived"]), columns=["sex", "embarked"], drop_first=True
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

mlflow.set_experiment("titanic-random-forest")

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    # Guardar el modelo entrenado
    joblib.dump(model, "model.pkl")
    mlflow.sklearn.log_model(model, name="titanic-rf")

    # Registrar parametros y metricas
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", str(max_depth))
    mlflow.log_param("n_samples", len(df))
    mlflow.log_param("n_features", X.shape[1])
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_macro", f1)

    print(
        f"n_estimators={n_estimators} max_depth={max_depth} "
        f"-> accuracy={accuracy:.4f} f1_macro={f1:.4f}"
    )
