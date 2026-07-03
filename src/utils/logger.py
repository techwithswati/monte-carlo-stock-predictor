"""
Structured JSON Logging
========================
PE-grade log format compatible with Datadog, ELK, and CloudWatch.
"""

import logging
import os
import sys
import json
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""
    
    RESERVED = {"message", "asctime", "levelname", "name", "pathname", "lineno"}
    
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        
        # Merge any extra kwargs passed to the logger
        for k, v in record.__dict__.items():
            if k not in logging.LogRecord.__dict__ and k not in self.RESERVED:
                try:
                    json.dumps(v)
                    payload[k] = v
                except (TypeError, ValueError):
                    payload[k] = str(v)
        
        return json.dumps(payload)


def setup_logging(level: str | None = None) -> None:
    log_level = level or os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "json").lower()
    
    root = logging.getlogger()
    root.setLevel(log_level)
    
    # Avoid duplicate handlers on re-import
    if root.handlers:
        return
    
    handler = logging.StreamHandler(sys.stdout)
    
    if log_format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatting("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        
    root.addHandler(handler)
    
    # Silence noisy third-party loggers
    for noisy in ("urllib3", "yfinance", "peewee", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
