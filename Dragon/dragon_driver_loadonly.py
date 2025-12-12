import os
from time import perf_counter
import argparse
from typing import List
import shutil
import pathlib
import dragon
from math import ceil
import multiprocessing as mp
from dragon.data.ddict.ddict import DDict
from dragon.native.machine import System, Node
from dragon.infrastructure.policy import Policy

from data_loader.data_loader_presorted import load_inference_data
from data_loader.data_loader_presorted import get_files

def parseNodeList() -> List[str]:
    """
    Parse the node list provided by the scheduler

    :return: PBS node list
    :rtype: list 
    """
    nodelist = []
    hostfile = os.getenv('PBS_NODEFILE')
    with open(hostfile) as file:
        nodelist = file.readlines()
        nodelist = [line.rstrip() for line in nodelist]
        nodelist = [line.split('.')[0] for line in nodelist]
    return nodelist

if __name__ == "__main__":
    
    # Import command line arguments
    parser = argparse.ArgumentParser(description='Distributed dictionary example')
    parser.add_argument('--data_dictionary_mem_fraction', type=float, default=0.7,
                        help='fraction of memory dedicated to data dictionary')
    parser.add_argument('--inference_node_num', type=int, default=1,
                        help='number of nodes running inference')
    parser.add_argument('--sorting_node_num', type=int, default=1,
                        help='number of nodes running sorting')
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
    parser.add_argument('--only_load_data', type=int, default=0,
                        help='Flag to only do the data loading')
    args = parser.parse_args()

    # Start driver
    start_time = perf_counter()
    print("Begun dragon driver", flush=True)
    print(f"Reading inference data from path: {args.data_path}", flush=True)
    mp.set_start_method("dragon")

    # Get information about the allocation
    pbs_nodelist = parseNodeList()
    alloc = System()
    num_tot_nodes = int(alloc.nnodes)
    tot_nodelist = alloc.nodes

    # for this sequential loop test set inference and docking to all the nodes and sorting and training to one node
    node_counts = {"sorting": 1, 
                    "training": 1, 
                    "inference": num_tot_nodes,
                    "docking": num_tot_nodes}
    
    nodelists = {}
    offset = 0
    for key in node_counts.keys():
        nodelists[key] = tot_nodelist[:node_counts[key]]
          
    # Get info on the number of files
    base_path = pathlib.Path(args.data_path)
    files, num_files = get_files(base_path)

    mem_per_file = 12/8192
    tot_mem = int(min(args.mem_per_node*num_tot_nodes,
                  max(ceil(num_files*mem_per_file*100/args.data_dictionary_mem_fraction),2*num_tot_nodes)
                  ))
    print(f"There are {num_files} files, setting mem_per_node to {tot_mem/num_tot_nodes}")

    # Set up and launch the inference data DDict and top candidate DDict
    data_dict_mem = max(int(args.data_dictionary_mem_fraction*tot_mem), num_tot_nodes)
    candidate_dict_mem = max(int(tot_mem - data_dict_mem), num_tot_nodes)
    print(f"Setting data_dict size to {data_dict_mem} GB and candidate_dict size to {candidate_dict_mem} GB")
    data_dict_mem *= (1024*1024*1024)
    candidate_dict_mem *= (1024*1024*1024)

    # Start distributed dictionary used for inference
    #inf_dd_policy = Policy(placement=Policy.Placement.HOST_NAME, host_name=Node(inf_dd_nodelist).hostname)
    # Note: the host name based policy, as far as I can tell, only takes in a single node, not a list
    #       so at the moment we can't specify to the inf_dd to run on a list of nodes.
    #       But by setting inf_dd_nodes < num_tot_nodes, we can make it run on the first inf_dd_nodes nodes only
    data_dd = DDict(args.managers_per_node, num_tot_nodes, data_dict_mem, trace=True)
    print(f"Launched Dragon Dictionary for inference with total memory size {data_dict_mem}", flush=True)
    print(f"on {num_tot_nodes} nodes", flush=True)
    print(f"{data_dd.stats=}")
    
    # Launch the data loader component
    max_procs = args.max_procs_per_node*num_tot_nodes
    print("Loading inference data into Dragon Dictionary ...", flush=True)
    tic = perf_counter()
    loader_proc = mp.Process(target=load_inference_data, 
                             args=(data_dd, 
                                   args.data_path, 
                                   max_procs, 
                                   num_tot_nodes*args.managers_per_node),
                                   num_files,
                            )
    loader_proc.start()
    loader_proc.join()
    toc = perf_counter()
    load_time = toc - tic
    if loader_proc.exitcode == 0:
        print(f"Loaded inference data in {load_time:.3f} seconds", flush=True)
    else:
        raise Exception(f"Data loading failed with exception {loader_proc.exitcode}")

    cand_dd = DDict(args.managers_per_node, num_tot_nodes, candidate_dict_mem, policy=None, trace=True)
    print(f"Launched Dragon Dictionary for top candidates with total memory size {candidate_dict_mem}", flush=True)
    print(f"on {num_tot_nodes} nodes", flush=True)
    
    # Close the dictionary
    print("Closing the Dragon Dictionary and exiting ...", flush=True)
    cand_dd.destroy()
    data_dd.destroy()
    end_time = perf_counter()
    print(f"Total time {end_time - start_time} s", flush=True)



