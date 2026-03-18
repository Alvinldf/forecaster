import schedule
import time
import logging
from data_ingestion import run_ingestion
from train import run_training
from evaluate import get_latest_rmse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ForecasterScheduler")

# Hardcoded threshold for retraining
# If RMSE exceeds this, we trigger a new training run
RMSE_THRESHOLD = 50.0  

def daily_job():
    logger.info("--- Starting Daily Scheduled Job ---")
    
    # 1. Run Data Ingestion (Incremental)
    try:
        logger.info("Running incremental data ingestion...")
        run_ingestion()
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        return

    # 2. Evaluate Model
    try:
        logger.info("Evaluating current model performance...")
        latest_rmse = get_latest_rmse()
        
        if latest_rmse is None:
            logger.warning("No previous RMSE found. Triggering initial training...")
            run_training()
        elif latest_rmse > RMSE_THRESHOLD:
            logger.info(f"RMSE ({latest_rmse:.4f}) exceeds threshold ({RMSE_THRESHOLD}). Triggering retraining...")
            run_training()
        else:
            logger.info(f"Model is healthy (RMSE: {latest_rmse:.4f} <= {RMSE_THRESHOLD}). Skipping retraining.")
            
    except Exception as e:
        logger.error(f"Evaluation or training failed: {e}")

    logger.info("--- Daily Scheduled Job Complete ---")

# Schedule the job for 21:00 UTC (approx market close)
schedule.every().day.at("21:00").do(daily_job)

if __name__ == "__main__":
    logger.info("Forecaster Scheduler Service Started.")
    logger.info("Scheduled to run daily at 21:00 UTC.")
    
    # Run once on startup to ensure we have data/model if the database is fresh
    # This is a common pattern for "cold start" scenarios
    daily_job()
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute
