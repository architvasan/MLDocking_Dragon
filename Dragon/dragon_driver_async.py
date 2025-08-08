import os
import logging
from time import perf_counter, sleep
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
from sorter.sorter import sort_controller
from docking_sim.launch_docking_sim import launch_docking_sim
from training.launch_training import launch_training
from data_loader.data_loader_presorted import get_files
from data_loader.model_loader import load_pretrained_model
from driver_functions import max_data_dict_size, output_sims
from logging_config import driver_logger as logger


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

    logger.info("Begun dragon driver")
    logger.info(f"Reading inference data from path: {args.data_path}")

    mp.set_start_method("dragon")

    # Get information about the allocation
    alloc = System()
    num_tot_nodes = int(alloc.nnodes)
    tot_nodelist = alloc.nodes

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
    num_files = 128

    tot_mem = args.mem_per_node*num_tot_nodes
    logger.info(f"There are {num_files} files")

    # There are 3 dictionaries:
    # 1. data dictionary for inference
    # 2. simulation dictionary for docking simulation results
    # 3. model and candidate dictionary for training
    # The model and candidate dictionary will be checkpointed


    # Set up and launch the inference data DDict and top candidate DDict
    # Calculate memory allocation for each dictionary
    data_dict_mem, sim_dict_mem, model_list_dict_mem = max_data_dict_size(num_files, max_pool_frac=0.5)
    logger.info(f"Setting data_dict size to {data_dict_mem} GB")
    logger.info(f"Setting sim_dict size to {sim_dict_mem} GB")
    logger.info(f"Setting model_list_dict size to {model_list_dict_mem} GB")

    # Check if total memory required exceeds available memory
    if data_dict_mem + sim_dict_mem + model_list_dict_mem > tot_mem:
        logger.info(f"Sum of dictionary sizes exceed total mem: {data_dict_mem=} {sim_dict_mem=} {model_list_dict_mem=} {tot_mem=}")
        raise Exception("Not enough memory for DDicts")

    # Convert memory sizes to bytes
    data_dict_mem *= (1024 * 1024 * 1024)
    sim_dict_mem *= (1024 * 1024 * 1024)
    model_list_dict_mem *= (1024 * 1024 * 1024)

    # Initialize Dragon Dictionaries for inference, docking simulation, and model list
    data_dd = DDict(args.managers_per_node, num_tot_nodes, data_dict_mem)
    logger.info(f"Launched Dragon Dictionary for inference with total memory size {data_dict_mem} on {num_tot_nodes} nodes")
    sim_dd = DDict(args.managers_per_node, num_tot_nodes, sim_dict_mem)
    logger.info(f"Launched Dragon Dictionary for docking simulation with total memory size {sim_dict_mem} on {num_tot_nodes} nodes")
    model_list_dd = DDict(args.managers_per_node, num_tot_nodes, model_list_dict_mem, working_set_size=10, wait_for_keys=True)
    logger.info(f"Launched Dragon Dictionary for model list with total memory size {model_list_dict_mem} on {num_tot_nodes} nodes")

    # Load data into the data dictionary
    max_procs = args.max_procs_per_node * num_tot_nodes
    logger.info("Loading inference data into Dragon Dictionary ...")
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

    logger.info("Here are the stats after data loading...")
    logger.info("Data Dictionary stats:")
    logger.info(data_dd.stats)

    # Load pretrained model
    load_pretrained_model(model_list_dd)

    # Initialize simulated compounds list
    model_list_dd.bput('simulated_compounds', [])

    # Report Loading time
    toc = perf_counter()
    load_time = toc - tic
    if loader_proc.exitcode == 0:
        logger.info(f"Loaded inference data in {load_time:.3f} seconds")
    else:
        raise Exception(f"Data loading failed with exception {loader_proc.exitcode}")

    # Update driver log
    logger.info(f"# {load_time=}")
    num_keys = len(data_dd.keys())
    logger.info(f"# {num_keys=}")
    logger.info(f"# {num_files=}")
    
    
    # Number of top candidates to produce
    # The number of top candidates is set to 1000 for small tests and 10000 for larger tests
    if num_tot_nodes < 3:
        top_candidate_number = 1000
    else:
        top_candidate_number = 10000

    # Finished dictionary initialization
    logger.info("Finished initializing dictionaries")


    # New model event for training
    new_model_event = mp.Event()

    # Continue event for all processes
    continue_event = mp.Event()
    continue_event.set()

    # sychronization barrier for inference processes and training
    num_procs = num_gpus*node_counts["inference"]
    barrier = mp.Barrier(parties=num_procs) # num_procs -1 for inference + 1 for training = num_procs

    logger.info(f"Current checkpoint: {model_list_dd.checkpoint_id}")

    # Launch the data inference component
    
    logger.info(f"Launching inference with {num_procs} processes ...")
    if num_tot_nodes < 3:
        inf_num_limit = 8
        logger.info(f"Running small test on {num_tot_nodes}; limiting {inf_num_limit} keys per inference worker")
    else:
        inf_num_limit = None

    tic = perf_counter()
    inf_proc = mp.Process(
        target=launch_inference,
        args=(
            data_dd,
            model_list_dd,
            nodelists["inference"],
            num_procs,
            inf_num_limit,
            continue_event,
            new_model_event,
            barrier,
        ),
    )
    inf_proc.start()
    
    # Launch data sorter component
    logger.info(f"Launching sorting ...")

    random_number_fraction = 0
    sorter_proc = mp.Process(target=sort_controller,
                                args=(
                                    data_dd,
                                    top_candidate_number,
                                    args.max_procs_per_node,
                                    nodelists['sorting'],
                                    model_list_dd,
                                    random_number_fraction,
                                    continue_event,
                                    ),
                                )
    sorter_proc.start()

#     # Launch Docking Simulations
    logger.info(f"Launched Docking Simulations")
    num_procs = args.max_procs_per_node * node_counts["docking"]
    num_procs = min(num_procs, top_candidate_number//4)
    num_procs = max(num_procs, node_counts["docking"])
    dock_proc = mp.Process(
        target=launch_docking_sim,
        args=(sim_dd, 
            model_list_dd, 
            num_procs, 
            nodelists["docking"], 
            continue_event,),
    )
    dock_proc.start()
    

#     # Launch Training
    logger.info(f"Launched Fine Tune Training")
    tic = perf_counter()
    BATCH = 64
    EPOCH = 500
    train_proc = mp.Process(
        target=launch_training,
        args=(
            model_list_dd,
            nodelists["training"][0],  # training is always 1 node
            sim_dd,
            BATCH,
            EPOCH,
            continue_event,
            new_model_event,
            barrier,
        ),
    )
    train_proc.start()

    # Monitor processes
    all_procs = [inf_proc, sorter_proc, dock_proc, train_proc]
    all_exitcodes = [proc.exitcode for proc in all_procs]
    all_procs_alive = True
    while all_procs_alive:
        for proc in all_procs:
            if not proc.is_alive():
                all_procs_alive = False
                logger.info(f"Process {proc.name} has exited with code {proc.exitcode}")
                break
        sleep(5)
    
    # Check if all processes exited successfully
    sleep(10)  # Give processes time to finish logging
    error_on_exit = False
    for proc in all_procs:
        if proc.exitcode != 0 and proc.exitcode is not None:
            error_on_exit = True
            logger.error(f"Process {proc.name} exited with code {proc.exitcode}")

    # If any process exited with an error code, raise an exception
    if error_on_exit:
        for proc in all_procs:
            proc.terminate()
        raise Exception("One or more processes exited with an error code")
    else:
        logger.info("All processes completed successfully")

    # If all proceesses completed successfully, join them
    train_proc.join()
    dock_proc.join()
    sorter_proc.join()
    inf_proc.join()
  