# logging_config.py
import logging
import os

from rich.logging import RichHandler

# def setup_logging():
#     logging.basicConfig(
#         level="INFO",
#         format="%(message)s",
#         datefmt="[%X]",
#         handlers=[RichHandler(rich_tracebacks=True)],
#     )


def setup_logging():
    """Simple Rich logging setup."""

    # Force colors
    os.environ["FORCE_COLOR"] = "1"

    # Clear existing handlers
    logging.getLogger().handlers.clear()

    # Simple Rich handler
    handler = RichHandler(
        show_time=True,
        omit_repeated_times=False,  # Show time on every log
        show_path=False,
    )

    # Configure logging
    logging.basicConfig(
        level="INFO", format="%(message)s", handlers=[handler], force=True
    )

    # Apply to ALL existing loggers (including vLLM)
    for name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.handlers = [handler]  # Replace all handlers
        logger.propagate = True
