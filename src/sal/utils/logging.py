# logging_config.py
import logging
import os

# def setup_logging():
#     logging.basicConfig(
#         level="INFO",
#         format="%(message)s",
#         datefmt="[%X]",
#         handlers=[RichHandler(rich_tracebacks=True)],
#     )


def setup_logging():
    """Initialise logging once: ensure a root StreamHandler at INFO and add colours."""

    os.environ["FORCE_COLOR"] = "1"

    # Ensure there is at least one handler and an INFO log level on the root logger
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        default_handler = logging.StreamHandler()
        default_handler.setLevel(logging.INFO)
        default_handler.setFormatter(
            logging.Formatter("%(levelname)s | %(name)s | %(message)s")
        )
        root_logger.addHandler(default_handler)
    # Even if handlers already exist, make sure INFO and above are emitted
    root_logger.setLevel(logging.INFO)

    class ColorFormatter(logging.Formatter):
        COLORS = {
            "INFO": "\033[34m",
            "WARNING": "\033[33m",
            "ERROR": "\033[91m",
            "CRITICAL": "\033[95m",
        }

        def format(self, record):
            color = self.COLORS.get(record.levelname, "")
            formatted_msg = super().format(record)
            return f"{color}{formatted_msg}\033[0m"

    # Apply to all existing loggers
    for name in [""] + list(logging.Logger.manager.loggerDict):
        logger = logging.getLogger(name)
        for handler in logger.handlers:
            # Safely extract existing format string if present
            fmt = None
            if handler.formatter is not None and hasattr(handler.formatter, "_fmt"):
                fmt = handler.formatter._fmt  # type: ignore[attr-defined]
            handler.setFormatter(ColorFormatter(fmt))
