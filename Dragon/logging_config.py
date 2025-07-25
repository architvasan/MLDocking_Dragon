import logging

def setup_logger(name, log_file, level=logging.INFO):
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


driver_logger = setup_logger('driver', None, level=logging.INFO)
load_logger = setup_logger('loader', "data_loader.log", level=logging.INFO)
inf_logger = setup_logger('inf', "inference.log", level=logging.INFO)
sim_logger = setup_logger('sim', "simulation.log", level=logging.INFO)
train_logger = setup_logger('train', "training.log", level=logging.INFO)
sort_logger = setup_logger('sort', "sorting.log", level=logging.INFO)


