import logging
import time
import io

# Create in-memory buffer for logs
log_stream = io.StringIO()

def setup_logger():
    class MalaysiaFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            # Malaysia = UTC + 8 hours
            malaysia_time = time.gmtime(record.created + 8 * 3600)
            return time.strftime("%Y-%m-%d %H:%M:%S", malaysia_time)

    formatter = MalaysiaFormatter("[%(asctime)s] %(levelname)s - %(message)s")

    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Console handler (keeps printing to terminal)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Memory handler (captures logs for Streamlit UI)
    memory_handler = logging.StreamHandler(log_stream)
    memory_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, memory_handler],  # ✅ two handlers
        force=True
    )

    return log_stream
