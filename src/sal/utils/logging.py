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
    """Add colors only - don't touch formatting."""

    os.environ["FORCE_COLOR"] = "1"

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
            handler.setFormatter(
                ColorFormatter(
                    handler.formatter._fmt
                    if hasattr(handler.formatter, "_fmt")
                    else None
                )
            )
