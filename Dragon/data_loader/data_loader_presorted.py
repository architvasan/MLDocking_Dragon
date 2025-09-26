import pathlib
import gzip
from time import perf_counter
from typing import Tuple
import argparse
import os
import sys
import random
import logging

import dragon
import multiprocessing as mp
from dragon.data.ddict import DDict
from dragon.native.machine import current, System, Node
from dragon.infrastructure.policy import Policy
import traceback
from functools import partial

from logging_config import load_logger as logger

def get_files(base_p: pathlib.PosixPath) -> Tuple[list, int]:
    """Return the file paths

    :param base_p: file path to location of raw data
    :type base_p: pathlib.PosixPath
    :return: tuple with list of file paths and number of files
    :rtype: tuple
    """
    files = []
    file_count = 0

    smi_files = base_p.glob("**/*.smi")
    gz_files = base_p.glob("**/*.gz")
    smi_file_list = []
    for file in sorted(smi_files):
        fname = str(file).split("/")[-1].split(".")[0]
        smi_file_list.append(fname)
        files.append(file)
        file_count += 1
    for file in sorted(gz_files):
        fname = str(file).split("/")[-1].split(".")[0]
        if fname not in smi_file_list:
            files.append(file)
            file_count += 1
    return files, file_count


def read_smiles(file_tuple: Tuple[int, str, int]):
    """Read the smile strings from file

    :param file_path: file path to open
    :type file_path: pathlib.PosixPath
    """
    tic_start = perf_counter()
    sort_test = os.getenv("TEST_SORTING")

    try:
        me = mp.current_process()

        data_dict = me.stash["ddict"]

        file_index = file_tuple[0]
        manager_index = file_tuple[2]
        file_path = file_tuple[1]

        smiles = []
        tic = perf_counter()
        f_name = str(file_path).split("/")[-1]
        f_extension = str(file_path).split("/")[-1].split(".")[-1]
        if f_extension == "smi":
            with file_path.open() as f:
                for line in f:
                    smile = line.split("\t")[0]
                    smiles.append(smile)
        elif f_extension == "gz":
            with gzip.open(str(file_path), "rt") as f:
                for line in f:
                    smile = line.split("\t")[0]
                    smiles.append(smile)
        read_time = perf_counter() - tic

        inf_results = [0.0 for i in range(len(smiles))]
        if sort_test:
            inf_results = [random.uniform(8.0, 14.0) for i in range(len(smiles))]
        key = f"{manager_index}_{file_index}"

        smiles_size = sum([sys.getsizeof(s) for s in smiles])
        smiles_size += sum([sys.getsizeof(infr) for infr in inf_results])
        smiles_size += sys.getsizeof(f_name)
        smiles_size += sys.getsizeof(key)

        f_name_list = f_name.split(".gz")
        logname = f_name_list[0].split(".")[0] + f_name_list[1]

        # with open(f"{outfiles_path}/{logname}.out",'w') as f:
        #     f.write(f"Worker located on {current().hostname}\n")
        #     f.write(f"Read smiles from {f_name}, smiles size is {smiles_size}\n")


        #print(f"Now putting key {key}", flush=True)
        tic = perf_counter()
        data_dict[key] = {"f_name": f_name, 
                          "smiles": smiles, 
                          "inf": inf_results, 
                          "model_iter": -1}
        put_time = perf_counter() - tic

        #print(f"Finished putting key {key}", flush=True)
        # data_dict[key] = smiles
        # with open(f"{outfiles_path}/{logname}.out",'a') as f:
        #     f.write(f"Stored data in dragon dictionary\n")
        #     f.write(f"key is {key}")
        toc_end = perf_counter()
        run_time = toc_end - tic_start

        return smiles_size, run_time, read_time, put_time
    except Exception as e:
        try:
            tb = traceback.format_exc()
            msg = "Error while reading smiles data:\n%s\n Traceback:\n%s"%(e, tb)
            outfiles_path = "smiles_sizes"
            if not os.path.exists(outfiles_path):
                os.mkdir(outfiles_path)
            with open(f"{outfiles_path}/{logname}.out", "a") as f:
                f.write(f"key is {key}")
                f.write(f"Worker located on {current().hostname}\n")
                f.write(f"Read smiles from {f_name}, smiles size is {smiles_size}\n")
                f.write("Exception was: %s\n"%msg)
                f.write("Pool Stats:\n%s\n"%data_dict.stats)
            raise Exception(e)
        except Exception as ex:
            print("GOT EXCEPTION IN EXCEPTION")
            print(ex)


def initialize_worker(the_ddict):
        # Since we want each worker to maintain a persistent handle to the DDict,
        # attach it to the current/local process instance. Done this way, workers attach only
        # once and can reuse it between processing work items
        me = mp.current_process()
        me.stash = {}
        me.stash["ddict"] = the_ddict

def load_inference_data(_dict: DDict, 
                        data_path: str, 
                        max_procs: int, 
                        num_managers: int, 
                        num_files: int = None,
                        nodelist: list = None,
                        load_split_factor: int = 1,
                        chunksize: int = 1):
    """Load pre-sorted inference data from files and to Dragon dictionary

    :param _dict: Dragon distributed dictionary
    :type _dict: DDict
    :param data_path: path to pre-sorted data
    :type data_path: str
    :param max_procs: maximum number of processes to launch for loading
    :type max_procs: int
    """
    # Get list of files to read
    base_path = pathlib.Path(data_path)
    files, num_files_in_dir = get_files(base_path)
    logger.info(f"{num_files_in_dir=}")
    if num_files is None:
        num_files = num_files_in_dir
    else:
        files = files[0:num_files]
    logger.info(f"Number of files to read is {num_files}")

    # alloc = System()
    # num_nodes = int(alloc.nnodes)
    # if num_nodes <= 2:
    #     gpu_devices = os.getenv("GPU_DEVICES")
    #     if gpu_devices is not None:
    #         gpu_devices = gpu_devices.split(",")
    #         num_gpus = len(gpu_devices)
    #     else:
    #         num_gpus = 0
    #     num_files = num_nodes*num_gpus*8 # 8 files per gpu
    #     files = files[0:num_files]
    #     print(f"Only loading {num_files} files for {num_nodes} node test")

    file_tuples = [(i, f, i % num_managers) for i, f in enumerate(files)]

    num_procs = min(max_procs, num_files)
    logger.info(f"Number of pool procs is {num_procs} and chunk size is {chunksize}")
    
    #policy = None
    #process_per_policy = 1
    #if nodelist is not None:
    #    policy = [Policy(placement=Policy.Placement.HOST_NAME, host_name=Node(nodelist[node]).hostname) \
    #                for node in range(len(nodelist))]

    tic = perf_counter()
    if load_split_factor == 1:
        ticc = perf_counter()
        pool = mp.Pool(num_procs, 
                       initializer=initialize_worker, 
                       initargs=(_dict,))
        t_init = perf_counter() - ticc
        logger.info(f"Pool initialized")
        logger.info(f"Reading smiles for {num_files}")
        outputs = pool.imap_unordered(read_smiles, file_tuples, chunksize=chunksize)
        pool.close()
        pool.join()
    else:
        outputs = []
        for i in range(load_split_factor):
            iter_data_size = 0
            num_pool_procs = num_procs
            pool = mp.Pool(num_pool_procs, 
                        initializer=initialize_worker, 
                        initargs=(_dict,))
            #print(f"Pool initialized", flush=True)
            #print(f"Reading smiles for {num_files}", flush=True)

            num_files_per_pool = num_files // load_split_factor + 1
            logger.info(f"{num_pool_procs=} {num_files_per_pool=}")
            output = pool.imap_unordered(
                read_smiles,
                file_tuples[
                    i
                    * num_files_per_pool : min((i + 1) * num_files_per_pool, num_files)
                ],
                chunksize=chunksize
            )
            outputs.extend(output)
            #for out in outputs:
            #    iter_data_size += out[0]
            #    io_times.append(out[1])
            #    ddict_times.append(out[2])
            #iter_data_size = iter_data_size/(1024.*1024.*1024.)
            #print(f"Size of dataset is {iter_data_size} GB", flush=True)
            #total_data_size += iter_data_size
            #print(f"Mapped function complete", flush=True)
            #pool.close()
            #print(f"Pool closed", flush=True)
            #pool.join()
            #print(f"Pool joined", flush=True)
    load_time = perf_counter() - tic
    
    logger.info(f"Loaded inference data in {load_time} sec ({t_init=})")

    total_data_size = 0
    run_times = []
    io_times = []
    ddict_times = []
    for out in outputs:
        total_data_size += out[0]
        run_times.append(out[1])
        io_times.append(out[2])
        ddict_times.append(out[3])
    total_data_size = total_data_size/(1024.*1024.*1024.)
    logger.info(f"Total data read {total_data_size} GB")
    logger.info(f"Run times: avg={sum(run_times)/len(run_times)} sec, max={max(run_times)} sec")
    logger.info(f"IO times: avg={sum(io_times)/len(io_times)} sec, max={max(io_times)} sec")
    logger.info(f"DDict times: avg={sum(ddict_times)/len(ddict_times)} sec, max={max(ddict_times)} sec")
    

if __name__ == "__main__":
    # Import command line arguments
    parser = argparse.ArgumentParser(description="Distributed dictionary example")
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="number of nodes the dictionary distributed across",
    )
    parser.add_argument(
        "--managers_per_node",
        type=int,
        default=1,
        help="number of managers per node for the dragon dict",
    )
    parser.add_argument(
        "--total_mem_size",
        type=int,
        default=8,
        help="total managed memory size for dictionary in GB",
    )
    parser.add_argument(
        "--max_procs",
        type=int,
        default=10,
        help="Maximum number of processes in a Pool",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC-22-2D-smaller_files",
        help="Path to pre-sorted SMILES strings to load",
    )
    args = parser.parse_args()

    # Start distributed dictionary
    mp.set_start_method("dragon")
    total_mem_size = args.total_mem_size * (1024 * 1024 * 1024)
    dd = DDict(args.managers_per_node, args.num_nodes, total_mem_size, trace=True)
    print("Launched Dragon Dictionary \n", flush=True)

    # Launch the data loader
    print("Loading inference data into Dragon Dictionary ...", flush=True)
    tic = perf_counter()
    load_inference_data(dd, args.data_path, args.max_procs)
    toc = perf_counter()
    load_time = toc - tic
    print(f"Loaded inference data in {load_time:.3f} seconds \n", flush=True)

    # Close the dictionary
    print("Done, closing the Dragon Dictionary", flush=True)
    dd.destroy()
