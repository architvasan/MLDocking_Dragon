import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# import logging
# logger = logging.getLogger(__name__)
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

from data_loader.data_loader_presorted import load_inference_data
from inference.launch_inference import launch_inference
from sorter.sorter import sort_dictionary_pg, sort_dictionary
from docking_sim.launch_docking_sim import launch_docking_sim
from training.launch_training import launch_training
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
    parser.add_argument('--max_iter', type=int, default=10,
                        help='Maximum number of iterations')
    parser.add_argument('--dictionary_timeout', type=int, default=10,
                        help='Timeout for Dictionary in seconds')
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
    files, num_files = get_files(base_path)

    tot_mem = args.mem_per_node*num_tot_nodes
    print(f"There are {num_files} files")

    # Set up and launch the inference data DDict and top candidate DDict
    data_dict_mem, candidate_dict_mem = max_data_dict_size(num_files, max_pool_frac = 0.5)
    print(f"Setting data_dict size to {data_dict_mem} GB and candidate_dict size to {candidate_dict_mem} GB")

    if data_dict_mem + candidate_dict_mem > tot_mem:
        print(f"Sum of dictionary sizes exceed total mem: {data_dict_mem=} {candidate_dict_mem=} {tot_mem=}", flush=True)
        raise Exception("Not enough memory for DDicts")

    data_dict_mem *= (1024*1024*1024)
    candidate_dict_mem *= (1024*1024*1024)

    # Start distributed dictionary used for inference
    # inf_dd_policy = Policy(placement=Policy.Placement.HOST_NAME, host_name=Node(inf_dd_nodelist).hostname)
    # Note: the host name based policy, as far as I can tell, only takes in a single node, not a list
    #       so at the moment we can't specify to the inf_dd to run on a list of nodes.
    #       But by setting inf_dd_nodes < num_tot_nodes, we can make it run on the first inf_dd_nodes nodes only
    data_dd = DDict(
        args.managers_per_node, num_tot_nodes, data_dict_mem
    )  # , trace=True)
    print(
        f"Launched Dragon Dictionary for inference with total memory size {data_dict_mem}",
        flush=True,
    )
    print(f"on {num_tot_nodes} nodes", flush=True)

    # Launch the data loader component
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

    print("Here are the stats after data loading...")
    print("++++++++++++++++++++++++++++++++++++++++")
    print(data_dd.stats)

    # Load pretrained model
    load_pretrained_model(data_dd)

    # Report Loading time
    toc = perf_counter()
    load_time = toc - tic
    if loader_proc.exitcode == 0:
        print(f"Loaded inference data in {load_time:.3f} seconds", flush=True)
    else:
        raise Exception(f"Data loading failed with exception {loader_proc.exitcode}")

    # Update driver log
    with open("driver_times.log", "a") as f:
        f.write(f"# {load_time=}\n")
    num_keys = len(data_dd.keys())
    with open("driver_times.log", "a") as f:
        f.write(f"# {num_keys=}\n")
    
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

    # Start sequential loop
    max_iter = args.max_iter
    iter = 0
    with open("driver_times.log", "a") as f:
        f.write(f"# iter  infer_time  sort_time  dock_time  train_time \n")
    while iter < max_iter:
        print(f"*** Start loop iter {iter} ***")
        iter_start = perf_counter()

        # Launch the data inference component
        num_procs = num_gpus*node_counts["inference"]
        print(f"Launching inference with {num_procs} processes ...", flush=True)
        if num_tot_nodes < 3:
            inf_num_limit = 8
            print(f"Running small test on {num_tot_nodes}; limiting {inf_num_limit} keys per inference worker")
        else:
            inf_num_limit = None

        tic = perf_counter()
        inf_proc = mp.Process(
            target=launch_inference,
            args=(
                data_dd,
                nodelists["inference"],
                num_procs,
                inf_num_limit,
            ),
        )
        inf_proc.start()
        inf_proc.join()
        toc = perf_counter()
        infer_time = toc - tic
        print(f"Performed inference in {infer_time:.3f} seconds \n", flush=True)

        if inf_proc.exitcode != 0:
            raise Exception("Inference failed!\n")
        
        # Launch data sorter component
        print(f"Launching sorting ...", flush=True)
        tic = perf_counter()
        if iter == 0:
            cand_dd["max_sort_iter"] = "-1"

        random_number = int(0.1*top_candidate_number)
        print(f"Adding {random_number} random candidates to training", flush=True)
        if os.getenv("USE_MPI_SORT"):
            print("Using MPI sort",flush=True)
            max_sorter_procs = args.max_procs_per_node*node_counts["sorting"]
            sorter_proc = mp.Process(target=sort_dictionary_pg, 
                                     args=(data_dd,
                                           top_candidate_number,
                                           max_sorter_procs, 
                                           nodelists["sorting"],
                                           cand_dd,
                                           random_number,
                                           ),
                                    )
            sorter_proc.start()
            sorter_proc.join()
        else:
            print("Using filter sort", flush=True)
            sorter_proc = mp.Process(target=sort_dictionary,
                                      args=(
                                            data_dd,
                                            top_candidate_number,
                                            cand_dd,
                                            True,
                                            ),
                                      )
            sorter_proc.start()
            sorter_proc.join()
        if sorter_proc.exitcode != 0:
            raise Exception("Sorting failed\n")

        toc = perf_counter()
        sort_time = toc - tic
        print(f"Performed sorting of {num_keys} keys in {sort_time:.3f} seconds \n", flush=True)

        # Launch Docking Simulations
        print(f"Launched Docking Simulations", flush=True)
        tic = perf_counter()
        num_procs = args.max_procs_per_node * node_counts["docking"]
        num_procs = min(num_procs, top_candidate_number//4)
        dock_proc = mp.Process(
            target=launch_docking_sim,
            args=(cand_dd, iter, num_procs, nodelists["docking"]),
        )
        dock_proc.start()
        dock_proc.join()
        toc = perf_counter()
        dock_time = toc - tic
        #os.rename("finished_run_docking.log", f"finished_run_docking_{iter}.log")
        print(f"Performed docking in {dock_time:.3f} seconds \n", flush=True)
        
        print(f"Candidate Dictionary stats:", flush=True)
        print(cand_dd.stats)
        if dock_proc.exitcode != 0:
            raise Exception("Docking sims failed\n")

        # Launch Training
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
        iter_end = perf_counter()
        iter_time = iter_end - iter_start
        print(
            f"Performed iter {iter} in {iter_time} seconds \n", flush=True
        )
        with open("driver_times.log", "a") as f:
            f.write(f"{iter}  {infer_time}  {sort_time}  {dock_time}  {train_time}\n")

        tic = perf_counter()
        output_sims(cand_dd, iter=iter)
        toc = perf_counter()
        print(f"Output candidates in {toc -tic} seconds",flush=True)
        iter += 1


    # Close the dictionary
    print("Closing the Dragon Dictionary and exiting ...", flush=True)
    # cand_dd.destroy()
    data_dd.destroy()
    end_time = perf_counter()
    print(f"Total time {end_time - start_time} seconds", flush=True)
