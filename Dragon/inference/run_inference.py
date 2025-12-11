import os
import sys
from typing import List
import numpy as np
import psutil
import os
from time import perf_counter, sleep
import random
import gc
import socket
from tqdm import tqdm
from dragon.utils import host_id
import logging

import intel_extension_for_tensorflow as itex
from inference.utils_transformer import pad
from inference.utils_encoder import SMILES_SPE_Tokenizer
from data_loader.model_loader import retrieve_model_from_dict, load_pretrained_model
from logging_config import inf_logger as logger
from logging_config import driver_logger
from logging_config import setup_logger
import keras
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


driver_path = os.getenv("DRIVER_PATH")


def split_dict_keys(keys: List[str], size: int, proc: int) -> List[str]:
    """Read the keys containing inference data from the Dragon Dictionary
    and split equally among the procs

    :param keys: list of keys in the dictionary
    :type keys: List[str]
    :param size: Number of total procs
    :type size: int
    :param proc: Local proc ID
    :type proc: int
    :return: list of strings containing the split keys
    :rtype: List[str]
    """
    num_keys = len(keys)
  
    keys_per_proc = num_keys // size
    remainder = num_keys % size

    next_start_index = 0
    for i in range(proc+1):
        start_index = next_start_index
        end_index = min(start_index + keys_per_proc + (1 if i < remainder else 0),
                        num_keys)
        next_start_index = end_index
    split_keys = keys[start_index:end_index]
   
    random.shuffle(split_keys)
    return split_keys


def process_inference_data(hyper_params: dict, tokenizer, smiles_raw: List[str]):
    """Preprosess the raw SMILES strings to generate the model input data

    :param hyper_params: dictionary with the model hyperparameters
    :type hyper_params: dict
    :param tokenizer: tokenizer to be used for preprocessing
    :type tokenizer: ...
    :param smiles_raw: list of the raw smiles read from file or dict
    :type smiles_raw: list
    :return: model input data
    :rtype: ...
    """
    maxlen = hyper_params["tokenization"]["maxlen"]
    x_inference = np.array(
        [list(pad(tokenizer(smi)["input_ids"], maxlen, 0)) for smi in smiles_raw]
    )
    return x_inference


def load_model(new_model_event, i):
    """Check if the model has been updated and load it if so
    :param new_model_event: event to check if the model has been updated
    :type new_model_event: threading.Event
    :param i: current key index
    :type i: int
    :return: True if the model should be loaded, False otherwise
    :rtype: bool
    """
    # If new_model_event is not None, it means we are running the async workflow
    if new_model_event is not None:
        return new_model_event.is_set()
    else:
        return False

def continue_inference(finished_keys, continue_event):
    """Check if inference should continue
    :param finished_keys: boolean indicating if all keys have been processed
    :type finished_keys: bool
    :param continue_event: event to check if inference should continue
    :type continue_event: threading.Event
    :return: True if inference should continue, False otherwise
    :rtype: bool
    """
    serial_workflow = continue_event is None
    if serial_workflow:
        return not finished_keys
    else:
        return continue_event.is_set()

def get_local_keys(data_dd, proc: int, num_procs: int) -> List[str]:
    """Read the keys containing inference data from the Dragon Dictionary
    and split equally among the procs
    :param data_dd: Dragon Dictionary containing the inference data
    :type data_dd: dragon.Dictionary
    :param proc: local proc ID
    :type proc: int
    :param num_procs: total number of procs
    :type num_procs: int
    :return: list of strings containing the split keys
    :rtype: List[str]
    """
    # Get local keys
    current_host = host_id()
    manager_nodes = data_dd.manager_nodes
    keys = []
    #worker_logger.debug(f"{current_host=}")
    if proc == 0:
        logger.debug(f"{manager_nodes=}")
    for i in range(len(manager_nodes)):
        if manager_nodes[i].h_uid == current_host:
            local_manager = i
            #print(f"{proc}: getting keys from local manager {local_manager}")
            dm = data_dd.manager(i)
            keys.extend(dm.keys())
    #worker_logger.debug(f"{proc}: found {len(keys)} local keys")

    # Split keys in Dragon Dict    
    keys = [k for k in keys if "iter" not in k and "model" not in k]
    keys.sort()
    #print(f"{proc}: splitting keys over {num_procs} local procs")
    if num_procs > 1:
        split_keys = split_dict_keys(keys, num_procs, proc%num_procs)
    else:
        split_keys = keys
    #print(f"{proc}: {split_keys}",flush=True)
    #worker_logger.debug(f"Running inference on {len(split_keys)} keys")
    return split_keys

def get_tokenizer():
    # Set up tokenizer
    vocab_file = driver_path + "inference/VocabFiles/vocab_spe.txt"
    spe_file = driver_path + "inference/VocabFiles/SPE_ChEMBL.txt"
    tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file=spe_file)
    return tokenizer

def process_key(key, val, model, model_iter, hyper_params, tokenizer, data_dd, cutoff=9):

    inf_tic = perf_counter()
    BATCH = hyper_params["general"]["batch_size"]      
    smiles_raw = val["smiles"]
    x_inference = process_inference_data(hyper_params, tokenizer, smiles_raw)
    output = model.predict(x_inference, batch_size=BATCH, verbose=0).flatten()
    inf_toc = perf_counter()

    sort_index = np.flip(np.argsort(output)).tolist()
    smiles_sorted = [smiles_raw[i] for i in sort_index]
    pred_sorted = [
        (
            output[i].item()
            if output[i] > cutoff
            else 0.0
        )
        for i in sort_index
    ]

    val["smiles"] = smiles_sorted
    val["inf"] = pred_sorted
    val["model_iter"] = model_iter

    dd_write_tic = perf_counter()
    data_dd[key] = val
    dd_write_toc = perf_counter()
    return {'inference_time': inf_toc - inf_tic,
            'dd_write_time': dd_write_toc - dd_write_tic}

def run_inference_loop(model_list_dd,
                       data_dd,
                       proc: int,
                       num_procs: int,
                       continue_event=None,
                       new_model_event=None,
                       barrier=None,
                       ):
    '''Run inference loop
    :param model_list_dd: Dragon Dictionary containing the models
    :type model_list_dd: dragon.Dictionary
    :param data_dd: Dragon Dictionary containing the inference data
    :type data_dd: dragon.Dictionary
    :param proc: local proc ID
    :type proc: int
    :param num_procs: total number of procs
    :type num_procs: int
    :param continue_event: event to check if inference should continue
    :type continue_event: threading.Event
    :param new_model_event: event to check if the model has been updated
    :type new_model_event: threading.Event
    :param barrier: barrier to synchronize the procs
    :type barrier: threading.Barrier
    '''

    os.makedirs("inference_worker_logs", exist_ok=True)
    worker_logger = setup_logger(f'inf_worker_{proc}', f"inference_worker_logs/inference_worker_{proc}.log", level=logging.DEBUG)
    worker_logger.info(f"Starting inference worker {proc} of {num_procs} procs")

    # Set up model and tokenizer
    tokenizer = get_tokenizer()
    model,hyper_params = retrieve_model_from_dict(model_list_dd, 
                                                checkpoint=False)
    checkpoint_id = model_list_dd.checkpoint_id

    worker_logger.info(f"Retrieved model checkpoint {checkpoint_id} and tokenizer")
    # Get keys to process
    my_keys = get_local_keys(data_dd, proc, num_procs)
    num_keys = len(my_keys)
    worker_logger.info(f"Processing {num_keys} keys")

    # Loop over keys until all keys are processed or continue_event is cleared
    next_key_index = 0
    while continue_inference(next_key_index < num_keys,
                            continue_event):

        # Get current model iteration
        checkpoint_id = model_list_dd.checkpoint_id

        tic = perf_counter()
        # Retrieve next key and its value
        next_key_index %= num_keys
        this_key = my_keys[next_key_index]

        dd_read_tic = perf_counter()
        val = data_dd[this_key]
        dd_read_toc = perf_counter()
        dd_read_time = dd_read_toc - dd_read_tic

        # Check if key value is stale
        # If yes, update model and continue
        # Loop will wait here until new model is available
        if val['model_iter'] == checkpoint_id:
            worker_logger.info(f"Key {this_key} is stale, waiting for new model")
            model = update_model(model_list_dd, 
                                hyper_params, 
                                new_model_event, 
                                barrier, 
                                proc,
                                worker_logger)
            worker_logger.info(f"New model loaded")
        
        # Process key
        metrics = process_key(this_key, val, model, checkpoint_id, hyper_params, tokenizer, data_dd)
        toc = perf_counter()
        key_time = toc - tic
        inference_time = metrics['inference_time']
        dd_write_time = metrics['dd_write_time']
        worker_logger.debug(f"Processed key {this_key}: {key_time=} {inference_time=} {dd_read_time=} {dd_write_time=} seconds")
        # Move to next key and check for model update
        next_key_index += 1
        try:
            new_model = new_model_event.is_set()
        except:
            new_model = False
        if new_model:
            worker_logger.info(f"Getting new model")
            model = update_model(model_list_dd, 
                                hyper_params, 
                                new_model_event, 
                                barrier, 
                                proc,
                                worker_logger)

def update_model(model_list_dd, hyper_params, 
                           new_model_event, barrier, proc, worker_logger):
    '''Update the model when a new model is available
    :param model_list_dd: Dragon Dictionary containing the models
    :type model_list_dd: dragon.Dictionary
    :param hyper_params: dictionary with the model hyperparameters
    :type hyper_params: dict
    :param new_model_event: event to check if the model has been updated
    :type new_model_event: threading.Event
    :param barrier: barrier to synchronize the procs
    :type barrier: threading.Barrier
    :param proc: local proc ID
    :type proc: int
    :return: updated model
    :rtype: ...
    '''
    
    worker_logger.info(f"Waiting for new model event")
    new_model_event.wait()
    

    # Retrieve model from dictionary
    model,_ = retrieve_model_from_dict(model_list_dd, 
                                        checkpoint=True, 
                                        hyper_params=hyper_params)
    checkpoint_id = model_list_dd.checkpoint_id
    worker_logger.info(f"Detected new model event, now on model {checkpoint_id}")
    worker_logger.debug(f"Waiting for barrier sync")
    barrier.wait()
    logger.info(f"Barrier cleared, proceeding to inference")
    return model


## Run main
if __name__ == "__main__":
    
    import gzip
    import glob
    
    num_procs = 1
    proc = 0
    continue_event = None
    dd = {}
    model_list_dd = {}
    load_pretrained_model(model_list_dd)
    
    file_dir = os.getenv("DATA_PATH")
    all_files = glob.glob(file_dir+"*.gz")
    files = all_files[0:1]
    num_files = len(files)
    file_tuples = [(i,fpath,i) for i,fpath in enumerate(files)]

    for file_tuple in file_tuples:
        file_index = file_tuple[0]
        manager_index = file_tuple[2]
        file_path = file_tuple[1]
        
        smiles = []
        f_name = str(file_path).split("/")[-1]
        f_extension = str(file_path).split("/")[-1].split(".")[-1]
        if f_extension=="smi":
            with file_path.open() as f:
                for line in f:
                    smile = line.split("\t")[0]
                    smiles.append(smile)
        elif f_extension=="gz":
            with gzip.open(str(file_path), 'rt') as f:
                for line in f:
                    smile = line.split("\t")[0]
                    smiles.append(smile)

        inf_results = [0.0 for i in range(len(smiles))]
        key = f"{manager_index}_{file_index}"
        f_name_list = f_name.split('.gz')
        logname =  f_name_list[0].split(".")[0]+f_name_list[1]
        dd[key] = {"f_name": f_name, 
                   "smiles": smiles,
                   "inf": inf_results}
    
    run_inference_loop(model_list_dd,
                       dd,
                       proc,
                       num_procs,
    )
                       #dd, num_procs, proc, continue_event, limit=None)
