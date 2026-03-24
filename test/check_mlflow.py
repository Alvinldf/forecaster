import mlflow
try:
    tracking_uri = "http://localhost:5000"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    experiments = mlflow.search_experiments()
    for exp in experiments:
        print(f"Experiment: {exp.name} (ID: {exp.experiment_id})")
except Exception as e:
    print(f"Error: {e}")
