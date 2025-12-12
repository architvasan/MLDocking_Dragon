import os
#import logging
#logger = logging.getLogger(__name__)
from time import perf_counter
import argparse
import pathlib
import dragon
import multiprocessing as mp
from dragon.data.ddict import DDict
from dragon.native.machine import System, Node
from dragon.infrastructure.policy import Policy

from data_loader.data_loader_presorted import get_files
from data_loader.model_loader import load_pretrained_model
from training.launch_training import launch_training
from driver_functions import max_data_dict_size

driver_path = os.getenv("DRIVER_PATH")

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
    parser.add_argument('--max_iter', type=int, default=10,
                        help='Maximum number of iterations')
    parser.add_argument('--dictionary_timeout', type=int, default=10,
                        help='Timeout for Dictionary in seconds')
    parser.add_argument('--data_path', type=str, default="/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC-22-2D-smaller_files",
                        help='Path to pre-sorted SMILES strings to load')

    args = parser.parse_args()
    
    # Start driver
    start_time = perf_counter()
    print("Begun dragon driver",flush=True)
    print(f"Reading inference data from path: {args.data_path}",flush=True)
    mp.set_start_method("dragon")

    # Get information about the allocation
    alloc = System()
    num_tot_nodes = int(alloc.nnodes)
    tot_nodelist = alloc.nodes
    tot_mem = args.mem_per_node * num_tot_nodes

    # Get info about gpus and cpus
    gpu_devices = os.getenv("GPU_DEVICES").split(",")
    num_gpus = len(gpu_devices)

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
        nodelists[key] = tot_nodelist[:node_counts[key]]

    # Set the number of nodes the dictionary uses
    num_dict_nodes = num_tot_nodes

    # Get info on the number of files
    base_path = pathlib.Path(args.data_path)
    files, num_files = get_files(base_path)

    tot_mem = args.mem_per_node*num_tot_nodes
    print(f"There are {num_files} files")

    # Set up and launch the inference data DDict and top candidate DDict
    data_dict_mem, candidate_dict_mem = max_data_dict_size(num_files)
    print(f"Setting data_dict size to {data_dict_mem} GB and candidate_dict size to {candidate_dict_mem} GB")
    
    if data_dict_mem + candidate_dict_mem > tot_mem:
        print(f"Sum of dictionary sizes exceed total mem: {data_dict_mem=} {candidate_dict_mem=} {tot_mem=}", flush=True)
        raise Exception("Not enough memory for DDicts")

    data_dict_mem *= (1024*1024*1024)
    candidate_dict_mem *= (1024*1024*1024)

    data_dd = DDict(args.managers_per_node, num_dict_nodes, data_dict_mem, trace=True)
    print(f"Launched Dragon Dictionary for inference with total memory size {data_dict_mem}", flush=True)
    print(f"on {num_dict_nodes} nodes", flush=True)
    
    
    # Load pretrained model
    load_pretrained_model(data_dd)

    cand_dd = DDict(args.managers_per_node, num_dict_nodes, candidate_dict_mem, policy=None, trace=True)
    print(f"Launched Dragon Dictionary for top candidates with total memory size {candidate_dict_mem}", flush=True)
    print(f"on {num_dict_nodes} nodes", flush=True)
    cand_dd.bput('simulated_compounds', [])

    # Load test simulation data
    with open(f"{driver_path}/training/training_test.data", "r") as f:
        lines = f.readlines()
        
        simulated_compounds = []
        for line in lines:
            print(line)
            if line[0] == "#": 
                continue
            line_list = line.split()
            smiles = line_list[0]
            dock_score = float(line_list[1])
            cand_dd[smiles] = {'dock_score': dock_score}
            simulated_compounds.append(smiles)
        cand_dd.bput('simulated_compounds', simulated_compounds)
    
    # Run training module
    print(f"Launched Fine Tune Training", flush=True)
    tic = perf_counter()
    BATCH = 64
    EPOCH = 500
    train_proc = mp.Process(
        target=launch_training,
        args=(
            data_dd,
            nodelists["training"][0],  # training is always 1 node
            cand_dd,
            BATCH,
            EPOCH,
        ),
    )
    train_proc.start()
    train_proc.join()
    toc = perf_counter()
    train_time = toc - tic
    print(f"Performed training in {train_time} seconds \n", flush=True)
    if train_proc.exitcode != 0:
        raise Exception("Training failed\n")
    print(f"Training completed successfully", flush=True)
