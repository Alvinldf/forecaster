import schedule
import time
import logging
from data_ingestion import run_ingestion
from predict import run_prediction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ForecasterScheduler")

def daily_job():
    logger.info("--- Starting Daily Scheduled Job ---")
    
    # 1. Run Data Ingestion (Fetch latest daily closes from yfinance/BCRP)
    try:
        logger.info("Running incremental data ingestion...")
        run_ingestion()
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        return

    # 2. Run Prediction (Multi-Task Inference Engine Pipeline)
    try:
        logger.info("Generating Procurement Strategy forecasts for the Next Period...")
        run_prediction()
    except Exception as e:
        logger.error(f"Prediction inference failed: {e}")

    logger.info("--- Daily Scheduled Job Complete ---")

# Schedule the job for 21:00 UTC (approx market close)
schedule.every().day.at("21:00").do(daily_job)

if __name__ == "__main__":
    logger.info("🚀 Forecaster Scheduler Service Started.")
    logger.info("Scheduled to run daily at 21:00 UTC.")
    
    # Run once on startup to ensure we have fresh data/signals instantly
    logger.info("Running initial cold-start execution...")
    daily_job()
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute
