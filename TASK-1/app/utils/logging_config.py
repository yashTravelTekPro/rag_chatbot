import logging
import sys

def setup_logging(level=logging.INFO):
    """
    Configure logging for the application
    """
    # create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    return root_logger

def get_logger(name: str):
    """
    Get a logger instance for a module
    """
    return logging.getLogger(name)
