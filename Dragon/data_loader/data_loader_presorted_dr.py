import pathlib
import gzip
from time import perf_counter
from typing import Tuple
import argparse
import os
import sys
import random
try:
    import intel_extension_for_tensorflow as itex
except ImportError:
    pass
import dragon
import multiprocessing as mp
from dragon.data.ddict import DDict
from dragon.native.machine import current, System
import traceback

from functools import partial

class DataReader:

    def __init__(self, dir, 
                num_files: int = None,
                num_managers: int = 1):
        
        files, file_count = get_files(pathlib.Path(dir))
        if num_files is None:
            num_files = file_count
        else:
            files = files[0:num_files]

        file_tuples = [(i, f, i % num_managers) for i, f in enumerate(files)]

        self.dir = dir
        self.file_tuples = [ft for ft in file_tuples]
        self.idx = 0
        self.num_files = num_files
        self.total_files = file_count

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data_to_process = self.file_tuples[self.idx]
            self.idx += 1
            return data_to_process
        except IndexError:
            raise StopIteration

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
    tic = perf_counter()
    outfiles_path = f"smiles_sizes/{file_tuple[2]}"
    sort_test = os.getenv("TEST_SORTING")
    debug = True

    if debug:
        os.makedirs(outfiles_path, exist_ok=True)

    try:
        
        file_tic = perf_counter()
        # Read smiles from file
        file_index = file_tuple[0]
        manager_index = file_tuple[2]
        file_path = file_tuple[1]

        smiles = []
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
        file_toc = perf_counter()
        
        f_name_list = f_name.split(".gz")
        logname = f_name_list[0].split(".")[0] + f_name_list[1]
        if debug:
            with open(f"{outfiles_path}/{logname}.out",'a') as f:
                f.write(f"Worker located on {current().hostname}\n")
                f.write(f"Read smiles from {f_name} in {file_toc - file_tic}s\n")
                f.write(f"Number of smiles read: {len(smiles)}\n")

        inf_results = [0.0 for i in range(len(smiles))]
        if sort_test:
            inf_results = [random.uniform(8.0, 14.0) for i in range(len(smiles))]
        key = f"{manager_index}_{file_index}"

        smiles_size = sum([sys.getsizeof(s) for s in smiles])
        smiles_size += sum([sys.getsizeof(infr) for infr in inf_results])
        smiles_size += sys.getsizeof(f_name)
        smiles_size += sys.getsizeof(key)

        stash_tic = perf_counter()
        # Get handle to Dragon dictionary from worker process stash
        me = mp.current_process()
        data_dict = me.stash["ddict"]
        stash_toc = perf_counter()
        if debug:
            with open(f"{outfiles_path}/{logname}.out",'a') as f:
                f.write(f"Retrieved ddict from stash in {stash_toc - stash_tic}s\n")

        ddict_tic = perf_counter()
        #print(f"Now putting key {key}", flush=True)
        data_dict[key] = {"f_name": f_name, 
                          "smiles": smiles, 
                          "inf": inf_results, 
                          "model_iter": -1}
        ddict_toc = perf_counter()
        toc = perf_counter()
        if debug:
            with open(f"{outfiles_path}/{logname}.out",'a') as f:
                f.write(f"Stored {smiles_size} bytes in dragon dictionary in {ddict_toc - ddict_tic}s\n")
                f.write(f"Total time to read and store smiles from {f_name} is {toc - tic}s\n")
        return smiles_size
    except Exception as e:
        try:
            tb = traceback.format_exc()
            msg = "Error while reading smiles data:\n%s\n Traceback:\n%s"%(e, tb)
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
                        file_chunk_num: int = 32):
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
    #files, num_files_in_dir = get_files(base_path)
    dr = DataReader(base_path, num_files, num_managers)
    print(f"num_files_in_dir={dr.total_files}", flush=True)
    
    print(f"Number of files to read is {num_files}", flush=True)

    num_pool_procs = min(max_procs, num_files//4)
    print(f"Number of pool procs is {num_pool_procs}", flush=True)
    
    total_data_size = 0
    start_time = perf_counter()
    
    print(f"Reading smiles for {num_files}", flush=True)
    chunksize = 64 #num_files//num_pool_procs//2
    print(f"Using chunksize of {chunksize}", flush=True)
    total_data_size = 0
    pool = mp.Pool(num_pool_procs, initializer=initialize_worker, initargs=(_dict,))
    print(f"Pool initialized", flush=True)
    
    smiles_sizes = pool.imap_unordered(
        read_smiles,
        dr,
        chunksize=chunksize,
    )

    tic = perf_counter()
    for i, size in enumerate(smiles_sizes):
        toc = perf_counter()
        total_data_size += size
        print(f"Loaded file {i} of size {size} bytes in {toc - tic}s, total size loaded: {total_data_size/(1024.*1024.*1024.)}", flush=True)
        tic = perf_counter()
    

    print(f"Mapped function complete", flush=True)
    pool.close()
    print(f"Pool closed", flush=True)
    pool.join()
    print(f"Pool joined", flush=True)
    total_data_size /= (1024.*1024.*1024.)
    print(f"Total data read in {total_data_size} GB", flush=True)
    

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
