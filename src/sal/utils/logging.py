# logging_config.py
import logging
import os

from rich.console import Console
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

    # Create console with no wrapping
    console = Console(width=None, soft_wrap=False)

    # Simple Rich handler
    handler = RichHandler(
        console=console, show_time=True, omit_repeated_times=False, show_path=False
    )

    # Configure logging
    logging.basicConfig(
        level="INFO", format="%(message)s", handlers=[handler], force=True
    )

    # FORCE all loggers to INFO level (override any DEBUG settings)
    logging.getLogger().setLevel(logging.INFO)  # Root logger

    for name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)  # Force to INFO
        logger.disabled = False  # Make sure it's not disabled
        logger.propagate = True
