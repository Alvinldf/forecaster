import mlflow
from config import INFLUX_URL  # Not used here, but ensures config is loaded

def get_latest_rmse(experiment_name="Copper_SaaS_Final"):
    """
    Queries MLflow for the latest run in the specified experiment 
    and returns its RMSE metric.
    """
    try:
        # Note: In the scheduler container, this will point to http://mlflow:5000
        client = mlflow.tracking.MlflowClient()
        
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"Experiment '{experiment_name}' not found.")
            return None
            
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if not runs:
            print(f"No runs found for experiment '{experiment_name}'.")
            return None
            
        latest_run = runs[0]
        rmse = latest_run.data.metrics.get("rmse")
        
        return rmse
    except Exception as e:
        print(f"Error fetching RMSE from MLflow: {e}")
        return None

if __name__ == "__main__":
    # Test local
    print(f"Latest RMSE: {get_latest_rmse()}")
