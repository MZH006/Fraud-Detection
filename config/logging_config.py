import logging
import sys
from contextvars import ContextVar

from pythonjsonlogger import jsonlogger

request_id_var: ContextVar[str] = ContextVar("request_id", default="N/A")


class RequestIdFilter(logging.Filter):
    """Add request ID to log records from context"""

    def filter(self, record):
        record.request_id = request_id_var.get()
        return True


class ColoredFormatter(logging.Formatter):
    """Colored formatter for development logs"""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logging(
    log_level: str = "INFO", log_format: str = "json", enable_colors: bool = True
) -> logging.Logger:
    """Configure structured logging for the application."""

    if log_format == "json":
        log_fmt = "%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s"
        formatter = jsonlogger.JsonFormatter(log_fmt)
    else:
        log_fmt = "%(asctime)s [%(levelname)s] [%(request_id)s] %(name)s - %(message)s"
        if enable_colors and sys.stdout.isatty():
            formatter = ColoredFormatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        else:
            formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    logger = logging.getLogger(name)
    return logger


def set_request_id(request_id: str) -> None:
    """Set request ID in context for current async context."""
    request_id_var.set(request_id)


def get_request_id() -> str:
    """Get current request ID from context."""
    return request_id_var.get()
