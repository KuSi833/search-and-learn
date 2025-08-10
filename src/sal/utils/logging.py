# logging_config.py
import logging

from rich.logging import RichHandler

# def setup_logging():
#     logging.basicConfig(
#         level="INFO",
#         format="%(message)s",
#         datefmt="[%X]",
#         handlers=[RichHandler(rich_tracebacks=True)],
#     )


def setup_logging():
    """Force rich logging on ALL existing loggers."""
    # Remove all existing handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Set up rich handler
    rich_handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=True)

    # Configure root logger
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        handlers=[rich_handler],
        force=True,  # This is key!
    )

    # Apply to all existing loggers
    for name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.addHandler(rich_handler)
        logger.propagate = True
