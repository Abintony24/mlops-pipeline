import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()

versions = client.search_model_versions(
    "name='IrisRandomForest'"
)

for version in versions:
    print("Version:", version.version)
    print("Artifact source:", version.source)
    print("-------------------")