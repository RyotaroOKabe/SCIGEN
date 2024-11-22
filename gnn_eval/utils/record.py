import logging
from io import StringIO

# Initialize a StringIO buffer to store logs in memory
log_buffer = StringIO()
# Configure logging to write to both the console and the in-memory buffer
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),                # Logs to console
        logging.StreamHandler(log_buffer)      # Logs to in-memory buffer
    ]
)
logger = logging.getLogger(__name__)