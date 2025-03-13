import logging
import os

# Get the absolute path of the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Define the logs folder path inside the project root
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Ensure the logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# Define the log file path
LOG_FILE = os.path.join(LOGS_DIR, "mycotoxin_pipeline.log")

def get_logger(name):
    """Creates and returns a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create a file handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add the handler to the logger (avoid duplicates)
    if not logger.hasHandlers():
        logger.addHandler(file_handler)

    return logger
