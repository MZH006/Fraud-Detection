import logging
import sys
from typing import Optional
from pythonjsonlogger import jsonlogger
from contextvars import ContextVar

# Context variable for request ID tracking
request_id_var: ContextVar[str] = ContextVar('request_id', default='N/A')


class RequestIdFilter(logging.Filter):
    """Add request ID to log records from context"""

    def filter(self, record):
        record.request_id = request_id_var.get()
        return True


class ColoredFormatter(logging.Formatter):
    """Colored formatter for development logs"""

    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    enable_colors: bool = True
) -> logging.Logger:
    """
    Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format ('json' for production, 'text' for development)
        enable_colors: Enable colored output for text format (development only)

    Returns:
        Configured root logger instance

    Example:
        >>> logger = setup_logging(log_level="INFO", log_format="json")
        >>> logger.info("Application started", extra={"version": "2.0.0"})
    """

    # Create log format string
    if log_format == "json":
        # JSON format for production (machine-readable)
        log_fmt = '%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s'
        formatter = jsonlogger.JsonFormatter(log_fmt)
    else:
        # Human-readable format for development
        log_fmt = '%(asctime)s [%(levelname)s] [%(request_id)s] %(name)s - %(message)s'
        if enable_colors and sys.stdout.isatty():
            formatter = ColoredFormatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        else:
            formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')

    # Configure console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())

    # Configure root logger
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
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Logger instance with request ID filtering

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing transaction", extra={"tx_id": "123"})
    """
    logger = logging.getLogger(name)
    return logger


def set_request_id(request_id: str) -> None:
    """
    Set request ID in context for current async context.

    Args:
        request_id: Unique request identifier

    Example:
        >>> set_request_id("req-12345")
        >>> logger.info("Processing request")  # Will include request_id in logs
    """
    request_id_var.set(request_id)


def get_request_id() -> str:
    """
    Get current request ID from context.

    Returns:
        Current request ID or 'N/A' if not set
    """
    return request_id_var.get()
