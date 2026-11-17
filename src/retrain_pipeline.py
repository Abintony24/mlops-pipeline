import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from monitor_drift import check_data_drift


# MLflow configuration
MODEL_NAME = "IrisRandomForest"
ACCURACY_THRESHOLD = 0.90


def train_and_register_model():

    print("Initiating retraining process...")

    # 1. Load dataset
    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42
    )

    # 2. Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    # 3. Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Candidate Model Trained. Accuracy: {accuracy:.4f}")

    # 4. Check accuracy threshold
    if accuracy >= ACCURACY_THRESHOLD:

        print("Accuracy threshold satisfied. Registering model...")

        # 5. Log model in MLflow
        with mlflow.start_run(run_name="automated_retrain_run"):

            mlflow.log_param("trigger", "data_drift")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 5)

            mlflow.log_metric("accuracy", accuracy)

            # 6. Register model in MLflow Model Registry
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                registered_model_name=MODEL_NAME,
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree"
                ]
            )

            print("New model version registered successfully.")

            return model_info

    else:
        print("Accuracy below 90%. Model registration skipped.")

        return None


def run_pipeline():

    print("Checking production data for drift...")

    # 1. Detect data drift
    drift_detected = check_data_drift()

    # 2. Trigger retraining if drift is detected
    if drift_detected:

        print("Data drift confirmed! Triggering retraining pipeline...")

        train_and_register_model()

    else:

        print("No significant drift detected. Retraining skipped.")


if __name__ == "__main__":

    # Connect to MLflow database
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Set MLflow experiment
    mlflow.set_experiment("iris-continuous-retraining")

    # Execute pipeline
    run_pipeline()