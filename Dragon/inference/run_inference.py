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

#from inference.utils_transformer import ParamsJson, ModelArchitecture, pad
import intel_extension_for_tensorflow as itex
from inference.utils_transformer import pad
from inference.utils_encoder import SMILES_SPE_Tokenizer
#from training.ST_funcs.clr_callback import *
#from training.ST_funcs.smiles_regress_transformer_funcs import *
from data_loader.model_loader import retrieve_model_from_dict
from logging_config import inf_logger as logger
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

def continue_inference(continue_event, reanalysis_iter):
    """Check if inference should continue
    :param continue_event: event to check if inference should continue
    :type continue_event: threading.Event
    :param reanalysis_iter: current reanalysis iteration
    :type reanalysis_iter: int
    :return: True if inference should continue, False otherwise
    :rtype: bool
    """
    # If continue_event is None, it means we are running the sequenal workflow
    if continue_event is None:
        if reanalysis_iter == 0:
            return True 
        else:
            return False
    # If continue_event is not None, it means we are running the async workflow
    else:
        return continue_event.is_set()

def infer(data_dd, 
          model_list_dd, 
          num_procs, proc, 
          continue_event,
          checkpoint_id,
          limit=None, 
          debug=True,
          new_model_event=None,
          bar=None):
    """Run inference reading from and writing data to the Dragon Dictionary"""
    gc.collect()
    tic = perf_counter()
    # !!! DEBUG !!!

    os.makedirs("inference_worker_logs", exist_ok=True)
    worker_logger = setup_logger(f'inf_worker_{proc}', f"inference_worker_logs/inference_worker_{proc}.log", level=logging.DEBUG)
    worker_logger.info(f"Starting inference worker {proc} with {num_procs} procs")
    
    p = psutil.Process()
    core_list = p.cpu_affinity()
    
    logger.info(f"Opening inference worker log: inference_worker_logs/inference_worker_{proc}.log")
    worker_logger.debug(f"\n\nNew run")
        
    cuda_device = os.getenv("CUDA_VISIBLE_DEVICES")
    pvc_device = os.getenv("ZE_AFFINITY_MASK")
    device = None
    if cuda_device:
        device = cuda_device
    if pvc_device:
        device = pvc_device
    hostname = socket.gethostname()
    worker_logger.debug(f"Launching infer for worker {proc} from process {p} on core {core_list} on device {hostname}:{device}")

    
    # Get local keys
    current_host = host_id()
    manager_nodes = data_dd.manager_nodes
    keys = []
    worker_logger.debug(f"{current_host=}")
    if proc == 0:
        logger.debug(f"{manager_nodes=}")
    for i in range(len(manager_nodes)):
        if manager_nodes[i].h_uid == current_host:
            local_manager = i
            #print(f"{proc}: getting keys from local manager {local_manager}")
            dm = data_dd.manager(i)
            keys.extend(dm.keys())
    worker_logger.debug(f"{proc}: found {len(keys)} local keys")

    # Split keys in Dragon Dict    
    keys = [k for k in keys if "iter" not in k and "model" not in k]
    keys.sort()
    #print(f"{proc}: splitting keys over {num_procs} local procs")
    if num_procs > 1:
        split_keys = split_dict_keys(keys, num_procs, proc%num_procs)
    else:
        split_keys = keys
    #print(f"{proc}: {split_keys}",flush=True)
    worker_logger.debug(f"Running inference on {len(split_keys)} keys")
     
    
    # Set up tokenizer
    # if hyper_params['tokenization']['tokenizer']['category'] == 'smilespair':
    vocab_file = driver_path + "inference/VocabFiles/vocab_spe.txt"
    spe_file = driver_path + "inference/VocabFiles/SPE_ChEMBL.txt"
    tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file=spe_file)
    num_smiles = 0
    preproc_time = 0
    model_time = 0
    dictionary_time = 0
    data_moved_size = 0
    num_run = len(split_keys)
    if limit is not None:
        num_run = min(limit, num_run)
    # Iterate over keys in Dragon Dict
    
    cutoff = 9
    logger.info(f"worker {proc} processing {num_run} keys")
    reanalysis_iter = 0
    
    model,hyper_params = retrieve_model_from_dict(model_list_dd, 
                                                    checkpoint=False)
    model_iter = model_list_dd.checkpoint_id
    BATCH = hyper_params["general"]["batch_size"]

    completed_keys = []

    while continue_inference(continue_event, reanalysis_iter):
        worker_logger.debug(f"On reanalysis_iter {reanalysis_iter}\n")

        # If all keys have been processed, continue to next iteration
        if len(completed_keys) >= num_run and not load_model(new_model_event, reanalysis_iter):
            logger.info(f"worker {proc} has completed all {num_run} keys and there is no new model to load")
            sleep(10)
            continue

        # If there are unprocessed keys, loop through keys
        for ikey in range(num_run):
            key = split_keys[ikey]
            # If the model has been updated, load it
            if load_model(new_model_event, ikey+reanalysis_iter):
                # Reset completed keys
                completed_keys = []

                worker_logger.debug(f"{model_list_dd.checkpoint_id=}")

                # Retrieve model from dictionary
                model,_ = retrieve_model_from_dict(model_list_dd, 
                                                    checkpoint=True, 
                                                    hyper_params=hyper_params)
                model_iter = model_list_dd.checkpoint_id
                
                worker_logger.debug(f"Loaded model from checkpoint {model_iter}")
                if bar is not None and reanalysis_iter+ikey > 0:
                    worker_logger.debug(f"worker {proc} waiting for barrier sync")
                    bar.wait()
                else:
                    logger.info(f"worker {proc} proceeding to inference")
            
            # If the key has already been processed, skip it
            if key in completed_keys:
                continue
            
            ## Print progress to stdout every 8 iters
            #if ikey%8 == 0:
            #    logger.info(f"...worker {proc} has completed {ikey} keys out of {num_run} with model {model_iter}")
            if continue_inference(continue_event, reanalysis_iter):  # this check is to stop inference in async wf when model is retrained
                ktic = perf_counter()
                dict_tic = perf_counter()
                
                # print(f"worker {proc}: getting val from dd",flush=True)
                val = data_dd[key]
                # print(f"worker {proc}: finished getting val from dd",flush=True)
                
                dict_toc = perf_counter()
                key_dictionary_time = dict_toc - dict_tic

                key_data_moved_size = 0.
                for kkey in val.keys():
                    key_data_moved_size += sys.getsizeof(kkey)
                    if type(val[kkey]) == list:
                        key_data_moved_size += sum([sys.getsizeof(v) for v in val[kkey]]) 
                    else:
                        key_data_moved_size += sys.getsizeof(val[kkey])          

                smiles_raw = val["smiles"]
                x_inference = process_inference_data(hyper_params, tokenizer, smiles_raw)
                output = model.predict(x_inference, batch_size=BATCH, verbose=0).flatten()

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

                dict_tic = perf_counter()
                data_dd[key] = val
                dict_toc = perf_counter()
                key_dictionary_time += dict_toc - dict_tic

                for kkey in val.keys():
                    key_data_moved_size += sys.getsizeof(kkey)
                    if type(val[kkey]) == list:
                        key_data_moved_size += sum([sys.getsizeof(v) for v in val[kkey]]) 
                    else:
                        key_data_moved_size += sys.getsizeof(val[kkey]) 
                        
                num_smiles += len(smiles_sorted)

                ktoc = perf_counter()
                key_time = ktoc - ktic
                dictionary_time += key_dictionary_time
                data_moved_size += key_data_moved_size
                completed_keys.append(key)
                worker_logger.debug(
                            f"Performed inference on key {key} {key_time=} {len(smiles_sorted)=} {key_data_moved_size=} {key_dictionary_time=}\n"
                        )
            else:
                break
        reanalysis_iter += 1
        
    toc = perf_counter()

    metrics = {
        "num_smiles": num_smiles,
        "total_time": toc - tic,
        "data_move_time": dictionary_time,
        "data_move_size": data_moved_size,
    }
    logger.info(f"worker {proc} is all DONE in {toc - tic} seconds!! :)")
    logger.info(f"Performed inference on {num_run} files and {num_smiles} smiles: total={toc - tic}, IO={dictionary_time}, model={model_time}, preprocessing={preproc_time}")
    return metrics


## Run main
if __name__ == "__main__":
    
    import gzip
    import glob
    
    num_procs = 1
    proc = 0
    continue_event = None
    dd = {}


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
    
    infer(dd, num_procs, proc, continue_event, limit=None)
