import os
from time import perf_counter
import argparse
from typing import List
import random
import shutil
import pathlib
import dragon
from math import ceil
import multiprocessing as mp
from dragon.data.ddict import DDict
from dragon.native.machine import System, Node
from dragon.infrastructure.policy import Policy

from data_loader.data_loader_dummy import load_inference_data
#from inference.launch_inference import launch_inference
#from sorter.sorter import sort_dictionary_pg, sort_dictionary
#from docking_sim.launch_docking_sim import launch_docking_sim
#from training.launch_training import launch_training
from data_loader.data_loader_presorted import get_files
from data_loader.model_loader import load_pretrained_model
from driver_functions import max_data_dict_size, output_sims


if __name__ == "__main__":

    # Import command line arguments
    parser = argparse.ArgumentParser(description='Distributed dictionary example')
    parser.add_argument('--managers_per_node', type=int, default=1,
                        help='number of managers per node for the dragon dict')
    parser.add_argument('--mem_per_node', type=int, default=8,
                        help='managed memory size per node for dictionary in GB')
    parser.add_argument('--max_procs_per_node', type=int, default=10,
                        help='Maximum number of processes in a Pool')
    parser.add_argument('--num_files', type=int, default=128,
                        help='Number of files to load from the data path')
    parser.add_argument('--file_chunk_num', type=int, default=4,
                        help='Number of file chunks for loading')
    parser.add_argument('--data_path', type=str, default="/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC-22-2D-smaller_files",
                        help='Path to pre-sorted SMILES strings to load')

    args = parser.parse_args()

    # Start driver
    start_time = perf_counter()
    print("Begun dragon driver", flush=True)
    print(f"Reading inference data from path: {args.data_path}", flush=True)
    mp.set_start_method("dragon")

    with open("driver_times.log", "w") as f:
        f.write(f"# {args.data_path}\n")

    # Get information about the allocation
    alloc = System()
    num_tot_nodes = int(alloc.nnodes)
    tot_nodelist = alloc.nodes

    with open("driver_times.log", "a") as f:
        f.write(f"# {num_tot_nodes=}\n")

    tot_mem = args.mem_per_node * num_tot_nodes

    # Get info about gpus and cpus

    gpu_devices = os.getenv("GPU_DEVICES")
    if gpu_devices is not None:
        gpu_devices = gpu_devices.split(",")
        num_gpus = len(gpu_devices)
    else:
        num_gpus = 0

    # for this sequential loop test set inference and docking to all the nodes and sorting and training to one node
    node_counts = {
        "sorting": num_tot_nodes,
        "training": 1,
        "inference": num_tot_nodes,
        "docking": num_tot_nodes,
    }

    nodelists = {}
    offset = 0
    for key in node_counts.keys():
        nodelists[key] = tot_nodelist[: node_counts[key]]

    # Set the number of nodes the dictionary uses
    num_dict_nodes = num_tot_nodes

    # Get info on the number of files
    base_path = pathlib.Path(args.data_path)
    num_files = args.num_files
    
    tot_mem = args.mem_per_node*num_tot_nodes
    print(f"There are {num_files} files")

    # Set up and launch the inference data DDict and top candidate DDict
    data_dict_mem, candidate_dict_mem = max_data_dict_size(num_files, max_pool_frac = 0.5)
    print(f"Setting data_dict size to {data_dict_mem} GB and candidate_dict size to {candidate_dict_mem} GB")

    if data_dict_mem + candidate_dict_mem > tot_mem:
        print(f"Sum of dictionary sizes exceed total mem: {data_dict_mem=} {candidate_dict_mem=} {tot_mem=}", flush=True)
        #raise Exception("Not enough memory for DDicts")

    data_dict_mem *= (1024*1024*1024)
    candidate_dict_mem *= (1024*1024*1024)

    data_dd = DDict(args.managers_per_node, num_tot_nodes, data_dict_mem)  # , trace=True)
    print(f"Launched Dragon Dictionary for inference with total memory size {data_dict_mem}",
        flush=True,)
    print(f"on {num_tot_nodes} nodes", flush=True)

    # Launch the data loader component
    max_procs = min(args.max_procs_per_node, 8) * num_tot_nodes
    print("Loading inference data into Dragon Dictionary ...", flush=True)
    tic = perf_counter()
    loader_proc = mp.Process(
        target=load_inference_data,
        args=(
            data_dd,
            args.data_path,
            max_procs,
            num_tot_nodes * args.managers_per_node,
            num_files,
            args.file_chunk_num,
        ),
    )
    loader_proc.start()
    loader_proc.join()

    print("Here are the dictionary stats after data loading...")
    print("++++++++++++++++++++++++++++++++++++++++")
    for dm in data_dd.stats:
        print(f"{dm.manager_id=} {dm.num_keys=} {dm.total_bytes=} {dm.pool_utilization=}", flush=True)
    #print(data_dd.stats)

    # Load pretrained model
    load_pretrained_model(data_dd)

    # Report Loading time
    toc = perf_counter()
    load_time = toc - tic
    if loader_proc.exitcode == 0:
        print(f"Loaded inference data in {load_time:.3f} seconds, {loader_proc.exitcode}", flush=True)
    else:
        #print(f"Data loading failed with exception {loader_proc.exitcode}",flush=True)
        raise Exception(f"Data loading failed with exception {loader_proc.exitcode}")

    # Update driver log
    with open("driver_times.log", "a") as f:
        f.write(f"# {load_time=}\n")
    num_keys = len(data_dd.keys())
    with open("driver_times.log", "a") as f:
        f.write(f"# {num_keys=}\n")
    with open("driver_times.log", "a") as f:
        f.write(f"# {num_files=}\n")
    
    # Create candidate dictionary
    cand_dd = DDict(args.managers_per_node, num_dict_nodes, candidate_dict_mem, policy=None, trace=True)
    cand_dd.bput('simulated_compounds', [])
    cand_dd.bput('current_sort_iter', -1)
    
    print(f"Launched Dragon Dictionary for top candidates with total memory size {candidate_dict_mem}", flush=True)
    print(f"on {num_dict_nodes} nodes", flush=True)
    
    # Number of top candidates to produce
    if num_tot_nodes < 3:
        top_candidate_number = 1000
    else:
        top_candidate_number = 10000

    # Set up the inference and sorting processes
    iter = 0
    with open("driver_times.log", "a") as f:
        f.write(f"# iter  infer_time  sort_time mpi_sort_time \n")

    # Close the dictionary
    print("Closing the Dragon Dictionary and exiting ...", flush=True)
    cand_dd.destroy()
    data_dd.destroy()
    end_time = perf_counter()
    print(f"Total time {end_time - start_time} seconds", flush=True)
