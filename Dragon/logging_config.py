import os
import sys
import logging
from contextlib import contextmanager
os.environ['MLDOCKING_LOGGING_LEVEL'] = 'INFO'

mldocking_logging_level = os.environ['MLDOCKING_LOGGING_LEVEL'] = 'INFO'
if mldocking_logging_level == 'DEBUG':
    log_level = logging.DEBUG
elif mldocking_logging_level == 'INFO':
    log_level = logging.INFO
elif mldocking_logging_level == 'WARNING':
    log_level = logging.WARNING
elif mldocking_logging_level == 'ERROR':
    log_level = logging.ERROR
else:
    log_level = logging.INFO

# --- 1. Define the Logger Stream Handler ---
class LoggerWriter:
    """A file-like object that redirects output to a logger."""
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.buffer = ''

    def write(self, message):
        """Append message to buffer and log lines when newline is found."""
        self.buffer += message
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            if line.strip(): # Log non-empty lines
                self.logger.log(self.level, line.rstrip())

    def flush(self):
        """Log any remaining content in the buffer."""
        if self.buffer.strip():
            self.logger.log(self.level, self.buffer.rstrip())
        self.buffer = ''

# --- 2. Define a Context Manager for Redirection ---
@contextmanager
def stdout_to_logger(logger, level=logging.INFO):
    """Context manager to redirect sys.stdout to a specified logger."""
    old_stdout = sys.stdout
    sys.stdout = LoggerWriter(logger, level)
    try:
        yield
    finally:
        # Restore original stdout
        sys.stdout.flush()
        sys.stdout = old_stdout


def setup_logger(name, log_file, level=logging.DEBUG):
    """Function to set up a logger for a specific part of the program."""
    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
    if log_file is None:
        # If no log file is specified, use the console
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    # Prevent the logs from propagating to the root logger, which might have its own handlers
    logger.propagate = False
    return logger


driver_logger = setup_logger('driver', None, level=log_level)
load_logger = setup_logger('loader', "data_loader.log", level=log_level)
inf_logger = setup_logger('inf', "inference.log", level=log_level)
sim_logger = setup_logger('sim', "simulation.log", level=log_level)
train_logger = setup_logger('train', "training.log", level=log_level)
sort_logger = setup_logger('sort', "sorting.log", level=log_level)


