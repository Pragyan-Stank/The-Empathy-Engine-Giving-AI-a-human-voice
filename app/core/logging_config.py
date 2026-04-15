import logging
import sys
import uuid
import contextvars

request_id_var = contextvars.ContextVar("request_id", default="")

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True

def setup_logging():
    logger = logging.getLogger("empathy_engine")
    logger.setLevel(logging.INFO)
    
    # Exclude basic logs if logger is already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(name)s] - [ReqID: %(request_id)s] - %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(RequestIdFilter())
    
    logger.addHandler(console_handler)
    return logger

logger = setup_logging()
