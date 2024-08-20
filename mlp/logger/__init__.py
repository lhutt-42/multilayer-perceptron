"""
This module contains the logger configuration.
"""

import logging

from colorama import Fore, Style, init as colorama_init


def _setup_logger() -> logging.Logger:
    """
    Setup the logger.

    Returns:
        logging.Logger: The logger.
    """

    class Formatter(logging.Formatter):
        """
        Formatter for the logger.
        """

        def format(self, record: logging.LogRecord) -> str:
            """
            Format the log record.

            Args:
                record (logging.LogRecord): The log record to format.

            Returns:
                str: The formatted log record.
            """

            match (record.levelname):
                case 'CRITICAL':
                    record.levelname \
                        = f"{Fore.RED}{Style.BRIGHT}{record.levelname:8}{Style.RESET_ALL}"
                case 'ERROR':
                    record.levelname \
                        = f"{Fore.RED}{Style.BRIGHT}{record.levelname:8}{Style.RESET_ALL}"
                case 'WARNING':
                    record.levelname \
                        = f"{Fore.YELLOW}{Style.BRIGHT}{record.levelname:8}{Style.RESET_ALL}"
                case 'INFO':
                    record.levelname \
                        = f"{Fore.CYAN}{Style.BRIGHT}{record.levelname:8}{Style.RESET_ALL}"
                case _:
                    record.levelname \
                        = f"{Style.BRIGHT}{record.levelname:8}{Style.RESET_ALL}"

            return super().format(record)

    colorama_init(autoreset=True)

    new_logger = logging.getLogger(__name__)
    new_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = Formatter('%(levelname)-8s: %(message)s')
    console_handler.setFormatter(formatter)

    new_logger.addHandler(console_handler)

    return new_logger


logger = _setup_logger()

__all__ = [
    'logger'
]
