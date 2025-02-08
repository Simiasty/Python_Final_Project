import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("project.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)