import json
import logging
import os
import sys
from datetime import datetime, timezone

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage()
        }
        
        # Pull extra attributes if they exist
        for key in ["decision", "confidence", "outcome", "asset", "pnl", "kelly_fraction", "market_id", "platform"]:
            if hasattr(record, key):
                log_data[key] = getattr(record, key)
                
        return json.dumps(log_data)

def setup_logging(log_file: str = "logs/pipeline.log", log_level: int = logging.INFO):
    # Ensure directories exist
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # File handler for JSON structured logs
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(JSONFormatter())
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)
    
    # Console handler for human-readable output
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # Set third-party logs to warning
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    logging.info(f"Logging initialized. Writing JSON logs to: {log_file}")
