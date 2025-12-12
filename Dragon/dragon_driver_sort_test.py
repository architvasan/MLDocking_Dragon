import os
#import logging
#logger = logging.getLogger(__name__)
from time import perf_counter
import argparse
from typing import List
import shutil
import pathlib
import dragon
from math import ceil
import multiprocessing as mp
from dragon.data.ddict import DDict
from dragon.native.machine import System, Node
from dragon.infrastructure.policy import Policy

from data_loader.data_loader_presorted import load_inference_data, get_files
from sorter.sorter import sort_dictionary_pg
from data_loader.data_loader_presorted import get_files
#from inference.launch_inference import launch_inference

if __name__ == "__main__":

    # Import command line arguments
    parser = argparse.ArgumentParser(description='Distributed dictionary example')
    parser.add_argument('--data_dictionary_mem_fraction', type=float, default=0.7,
                        help='fraction of memory dedicated to data dictionary')
    parser.add_argument('--managers_per_node', type=int, default=1,
                        help='number of managers per node for the dragon dict')
    parser.add_argument('--mem_per_node', type=int, default=8,
                        help='managed memory size per node for dictionary in GB')
    parser.add_argument('--max_procs_per_node', type=int, default=10,
                        help='Maximum number of processes in a Pool')
    parser.add_argument('--data_path', type=str, default="/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC-22-2D-smaller_files",
                        help='Path to pre-sorted SMILES strings to load')

    args = parser.parse_args()
  
    print("Begun dragon driver",flush=True)
    print(f"Reading inference data from path: {args.data_path}",flush=True)
    mp.set_start_method("dragon")

    # Only testing sorting; set env var
    os.environ["TEST_SORTING"] = "1"

   # Get information about the allocation
    alloc = System()
    num_tot_nodes = int(alloc.nnodes)
    tot_nodelist = alloc.nodes

    # Get info on the number of files
    base_path = pathlib.Path(args.data_path)
    files, num_files = get_files(base_path)

    # Start distributed dictionary and load data
    tot_mem = num_tot_nodes*args.mem_per_node

    data_dict_mem = max(int(tot_mem), num_tot_nodes)
    candidate_dict_mem = max(int(tot_mem*(1.-args.data_dictionary_mem_fraction)), num_tot_nodes)
    print(f"Setting data_dict size to {data_dict_mem} GB and candidate_dict size to {candidate_dict_mem} GB")
    data_dict_mem *= (1024*1024*1024)
    candidate_dict_mem *= (1024*1024*1024)

    data_dd = DDict(args.managers_per_node, num_tot_nodes, data_dict_mem, trace=True)
    print(f"Launched Dragon Dictionary for inference with total memory size {data_dict_mem}", flush=True)
    print(f"on {num_tot_nodes} nodes", flush=True)

    max_procs = args.max_procs_per_node * num_tot_nodes
    print("Loading inference data into Dragon Dictionary ...", flush=True)
    tic = perf_counter()
    loader_proc = mp.Process(
        target=load_inference_data,
        args=(
            data_dd,
            args.data_path,
            max_procs,
            num_tot_nodes * args.managers_per_node,
            num_files
        ),
    )
    loader_proc.start()
    loader_proc.join()
    toc = perf_counter()
    load_time = toc - tic
    if loader_proc.exitcode == 0:
        print(f"Loaded inference data in {load_time:.3f} seconds", flush=True)
    else:
        raise Exception(f"Data loading failed with exception {loader_proc.exitcode}")
    loader_proc.close()
    tic = perf_counter()
    print("Here are the stats after data loading...")
    print("++++++++++++++++++++++++++++++++++++++++")
    print(data_dd.stats)
    toc = perf_counter()
    load_time = toc - tic
    print(f"Retrieved dictionary stats in {load_time:.3f} seconds", flush=True)

    cand_dd = DDict(args.managers_per_node, num_tot_nodes, candidate_dict_mem, policy=None, trace=True)
    print(f"Launched Dragon Dictionary for top candidates with total memory size {candidate_dict_mem}", flush=True)
    print(f"on {num_tot_nodes} nodes", flush=True)
    
    # Number of top candidates to produce
    if num_tot_nodes < 3:
        top_candidate_number = 1000
    else:
        top_candidate_number = 5000

    num_keys = len(data_dd.keys())

    # Sort
    print(f"Launching sorting ...", flush=True)
    tic = perf_counter()
    cand_dd["max_sort_iter"] = "-1"

    if os.getenv("USE_MPI_SORT"):
        print("Using MPI sort",flush=True)
        max_sorter_procs = args.max_procs_per_node*num_tot_nodes
        #max_sorter_procs = 1*num_tot_nodes
        sorter_proc = mp.Process(target=sort_dictionary_pg, 
                                    args=(data_dd,
                                        top_candidate_number, 
                                        max_sorter_procs, 
                                        tot_nodelist,
                                        cand_dd))
        sorter_proc.start()
        sorter_proc.join()
        sorter_proc.close()

    else:
        print("No other sorting method implemented",flush=True)
    toc = perf_counter()
    infer_time = toc - tic
    print(f"Performed sorting of {num_keys} keys in {infer_time:.3f} seconds \n", flush=True)
