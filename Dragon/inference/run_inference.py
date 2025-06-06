import os
import sys
from typing import List
import numpy as np
import psutil
import os
from time import perf_counter
import random
import gc
import socket
from tqdm import tqdm
from dragon.utils import host_id

#from inference.utils_transformer import ParamsJson, ModelArchitecture, pad
from inference.utils_transformer import pad
from inference.utils_encoder import SMILES_SPE_Tokenizer
#from training.ST_funcs.clr_callback import *
#from training.ST_funcs.smiles_regress_transformer_funcs import *
from data_loader.model_loader import retrieve_model_from_dict

import keras
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

driver_path = os.getenv("DRIVER_PATH")


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


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
    try:
        keys_per_proc = num_keys // size
        remainder = num_keys % size

        next_start_index = 0
        for i in range(proc+1):
            start_index = next_start_index
            end_index = min(start_index + keys_per_proc + (1 if i < remainder else 0),
                            num_keys)
            next_start_index = end_index
        split_keys = keys[start_index:end_index]
    except Exception as e:
        with open("error.out",'a') as f:
            f.write(f"Exception {e}\n")

    #if num_keys / size - num_keys // size > 0:
    #    num_keys_per_proc = num_keys // size + 1
    #else:
    #    num_keys_per_proc = num_keys // size
    #start_ind = proc * num_keys_per_proc
    #end_ind = (proc + 1) * num_keys_per_proc
    #if proc != (size - 1):
    #    split_keys = keys[start_ind:end_ind]
    #else:
    #    split_keys = keys[start_ind:]

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
    # Load model on first key of first analysis iteration always
    if i == 0:
        return True
    # If new_model_event is not None, it means we are running the async workflow
    if new_model_event is not None:
        return new_model_event.is_set()  

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
    # !!! DEBUG !!!
    if debug:
        p = psutil.Process()
        core_list = p.cpu_affinity()
        log_file_name = f"infer_worker_{proc}.log"
        print(f"Opening inference worker log {log_file_name}", flush=True)
        with open(log_file_name,'a') as f:
            f.write(f"\n\nNew run\n")
            f.write(f"Hello from process {p} on core {core_list}\n")
            f.flush()
        cuda_device = os.getenv("CUDA_VISIBLE_DEVICES")
        pvc_device = os.getenv("ZE_AFFINITY_MASK")
        device = None
        if cuda_device:
            device = cuda_device
        if pvc_device:
            device = pvc_device
        hostname = socket.gethostname()
        print(f"Launching infer for worker {proc} from process {p} on core {core_list} on device {hostname}:{device}", flush=True)
    
    
    # Get local keys
    current_host = host_id()
    manager_nodes = data_dd.manager_nodes
    keys = []
    print(f"{current_host=}",flush=True)
    if proc == 0:
        print(f"{manager_nodes=}",flush=True)
    for i in range(len(manager_nodes)):
        if manager_nodes[i].h_uid == current_host:
            local_manager = i
            #print(f"{proc}: getting keys from local manager {local_manager}")
            dm = data_dd.manager(i)
            keys.extend(dm.keys())
    print(f"{proc}: found {len(keys)} local keys")

    # Split keys in Dragon Dict    
    keys = [k for k in keys if "iter" not in k and "model" not in k]
    keys.sort()
    #print(f"{proc}: splitting keys over {num_procs} local procs")
    if num_procs > 1:
        split_keys = split_dict_keys(keys, num_procs, proc%num_procs)
    else:
        split_keys = keys
    #print(f"{proc}: {split_keys}",flush=True)
    if debug:
        with open(log_file_name, "a") as f:
            f.write(f"Running inference on {len(split_keys)} keys\n")
     
    
    # Set up tokenizer
    # if hyper_params['tokenization']['tokenizer']['category'] == 'smilespair':
    vocab_file = driver_path + "inference/VocabFiles/vocab_spe.txt"
    spe_file = driver_path + "inference/VocabFiles/SPE_ChEMBL.txt"
    tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file=spe_file)
    tic = perf_counter()
    num_smiles = 0
    dictionary_time = 0
    data_moved_size = 0
    num_run = len(split_keys)
    if limit is not None:
        num_run = min(limit, num_run)
    # Iterate over keys in Dragon Dict
    
    cutoff = 9
    print(f"worker {proc} processing {num_run} keys",flush=True)
    continue_event = None
    reanalysis_iter = 0
    
    while continue_inference(continue_event, reanalysis_iter):
        for ikey in range(num_run):
            if load_model(new_model_event, ikey+reanalysis_iter):
                # Load model from dictionary
                if debug:
                    with open(log_file_name, "a") as f:
                        f.write(f"{model_list_dd.current_checkpoint_id=}\n")
                # model_list_dd.sync_to_newest_checkpoint()

                model_list_dd._chkpt_id = checkpoint_id

                if debug:
                    with open(log_file_name, "a") as f:
                        f.write(f"{model_list_dd.current_checkpoint_id=}\n")

                # Retrieve model from dictionary
                model,hyper_params = retrieve_model_from_dict(model_list_dd)
                model_iter = model_list_dd.current_checkpoint_id
                BATCH = hyper_params["general"]["batch_size"]
                
                if debug:
                    with open(log_file_name, "a") as f:
                        f.write(f"Loaded model from checkpoint {model_iter}\n")
                if bar is not None and reanalysis_iter+ikey > 0:
                    print(f"worker {proc} waiting for barrier sync", flush=True)
                    bar.wait()
                else:
                    print(f"worker {proc} proceeding to inference", flush=True)
            # Print progress to stdout every 8 iters
            if ikey%8 == 0:
                print(f"...worker {proc} has completed {ikey} keys out of {num_run} with model {model_iter}", flush=True)
            if continue_inference(continue_event, reanalysis_iter):  # this check is to stop inference in async wf when model is retrained
                ktic = perf_counter()
                key = split_keys[ikey]
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

                if debug:
                    with open(log_file_name, "a") as f:
                        f.write(
                            f"Performed inference on key {key} {key_time=} {len(smiles_sorted)=} {key_data_moved_size=} {key_dictionary_time=}\n"
                        )
                    #print(
                    #    f"Performed inference on key {key} {key_time=} {len(smiles_sorted)=} {key_data_moved_size=} {key_dictionary_time=}",
                        #   flush=True,
                    #)
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
    print(f"worker {proc} is all DONE!! :)", flush=True)
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
