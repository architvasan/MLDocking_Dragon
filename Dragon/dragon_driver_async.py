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
from driver_functions import max_data_dict_size, get_available_threads, save_simulations
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
    parser.add_argument('--num_files', type=int, default=128,
                        help='Number of files to load')
    parser.add_argument('--data_path', type=str, default="/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC-22-2D-smaller_files",
                        help='Path to pre-sorted SMILES strings to load')

    args = parser.parse_args()

    # Start driver
    start_time = perf_counter()

    logger.info("Begun dragon driver")
    logger.info(f"Reading inference data from path: {args.data_path}")

    mp.set_start_method("dragon")

    # Get information about the allocation
    alloc = System()
    num_tot_nodes = int(alloc.nnodes)
    tot_nodelist = alloc.nodes

    # Get info about gpus and cpus
    gpu_devices = os.getenv("GPU_DEVICES")
    if gpu_devices is not None:
        gpu_devices = gpu_devices.split(",")
        num_gpus = len(gpu_devices)
    else:
        num_gpus = 0

    # for this test set inference and docking to all the nodes and sorting and training to one node
    node_counts = {
        "sorting": num_tot_nodes,
        "training": 1,
        "inference": num_tot_nodes,
        "simulation": num_tot_nodes,
    }

    nodelists = {}
    offset = 0
    for key in node_counts.keys():
        nodelists[key] = tot_nodelist[offset:offset+node_counts[key]]
        #offset += node_counts[key]
    
    logger.debug(f"{nodelists=}")

    # Set the number of nodes the dictionary uses
    num_dict_nodes = num_tot_nodes

    # There are 3 dictionaries:
    # 1. data dictionary for inference
    # 2. simulation dictionary for docking simulation results
    # 3. model and candidate dictionary for training
    # The model and candidate dictionary will be checkpointed
    
    # Get info on the number of files
    base_path = pathlib.Path(args.data_path)
    #files, num_files = get_files(base_path)
    num_files = args.num_files
    logger.info(f"There are {num_files} files")

    tot_mem = args.mem_per_node*num_tot_nodes
    
    # Set up and launch the inference data DDict and top candidate DDict
    # Calculate memory allocation for each dictionary
    data_dict_mem, sim_dict_mem, model_list_dict_mem = max_data_dict_size(num_files, node_counts, max_pool_frac=0.5)
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
    
    # Launch the data loader component
    max_procs = args.max_procs_per_node*num_tot_nodes
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
    
    logger.debug("Here are the data DDict stats after data loading...")
    for stat in data_dd.stats:
        logger.debug(f"manager_ID={stat.manager_id}, "+
                f"host_name={stat.hostname}, "+
                f"num_keys={stat.num_keys}, "+
                f"used_mem={stat.pool_utilization}, "+
                f"total_mem={stat.total_bytes}")
    
    # Number of top candidates to produce
    # The number of top candidates is set to 1000 for small tests and 10000 for larger tests
    if num_tot_nodes < 3:
        top_candidate_number = 1000
    else:
        top_candidate_number = 10000

    # Finished dictionary initialization
    logger.info("Finished initializing dictionaries")

    # Set number of processes for each component
    num_proc_sorting = 1 # this is the number of sorting instances, more processes can used by each instance
    num_proc_training = 1
    num_proc_inference = num_gpus * node_counts["inference"] - 1 # leave one gpu for training
    num_proc_simulation = args.max_procs_per_node * node_counts["simulation"]
    num_proc_simulation = min(num_proc_simulation, top_candidate_number//8)
    num_proc_simulation = max(num_proc_simulation, node_counts["simulation"])

    # Split threads not used by inference and training between sorting and simulation
    available_threads = get_available_threads()

    # Allocate 1 thread per node for sorting
    sorting_threads = [available_threads[0]]

    # Distribute remaining threads to simulations
    available_threads = available_threads[1:]
    num_proc_simulation = min(len(available_threads)*node_counts["simulation"], num_proc_simulation)
    spacing = max(len(available_threads)//(num_proc_simulation//node_counts["simulation"]), 1)
    simulation_threads = available_threads[::spacing][:num_proc_simulation]
    num_proc_simulation = len(simulation_threads)*node_counts["simulation"]

    logger.info(f"Number of inference workers: {num_proc_inference}")
    logger.info(f"Number of simulation workers: {num_proc_simulation}")
    logger.info(f"Number of sorting instances: {num_proc_sorting}")
    logger.info(f"Number of training instances: {num_proc_training}")

    logger.info(f"Number of threads for sorting per node: {len(sorting_threads)}")
    logger.info(f"Number of threads for simulation per node: {len(simulation_threads)}")

    # New model event for training
    new_model_event = mp.Event()

    # Continue event for all processes
    # continue_event = mp.Event()
    # continue_event.set()
    stop_event = mp.Event()

    # sychronization barrier for dictionary checkpoints
    num_parties = num_proc_simulation + num_proc_training + num_proc_inference + num_proc_sorting
    barrier = mp.Barrier(parties=num_parties) 
    logger.info(f"Barrier for new model created with {num_parties} parties")
    logger.info(f"Current checkpoint: {model_list_dd.checkpoint_id}")

    # Launch the data inference component
    logger.info(f"Launching inference with {num_proc_inference} processes ...")
    inf_num_limit = None
    if num_tot_nodes < 3:
        inf_num_limit = 8
        logger.info(f"Running small test on {num_tot_nodes} nodes; limiting {inf_num_limit} keys per inference worker")
    
    tic = perf_counter()
    inf_proc = mp.Process(
        target=launch_inference,
        args=(
            data_dd,
            model_list_dd,
            nodelists["inference"],
            num_proc_inference+1, # one process will be skipped to leave gpu for training
            inf_num_limit,
            #continue_event,
            stop_event,
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
                                    sorting_threads,
                                    model_list_dd,
                                    random_number_fraction,
                                    args.max_iter, # number of model retrainings
                                    #continue_event,
                                    stop_event,
                                    new_model_event,
                                    barrier,),
                                )
    sorter_proc.start()

#     # Launch Docking Simulations
    logger.info(f"Launching Docking Simulations...")
    dock_proc = mp.Process(
        target=launch_docking_sim,
        args=(sim_dd, 
            model_list_dd, 
            num_proc_simulation, 
            nodelists["simulation"],
            simulation_threads,
            stop_event,
            #continue_event,
            new_model_event,
            barrier),
    )
    dock_proc.start()
    

#     # Launch Training
    logger.info(f"Launching Fine Tune Training...")
    tic = perf_counter()
    BATCH = 64
    EPOCH = 150
    train_proc = mp.Process(
        target=launch_training,
        args=(
            model_list_dd,
            nodelists["training"][0],  # training is always 1 node
            sim_dd,
            BATCH,
            EPOCH,
            args.max_iter,
            stop_event,
            #continue_event,
            new_model_event,
            barrier,
        ),
    )
    train_proc.start()

    # Monitor processes
    logger.info(f"inf_proc is {inf_proc.name} with PID {inf_proc.pid}")
    logger.info(f"sorter_proc is {sorter_proc.name} with PID {sorter_proc.pid}")
    logger.info(f"dock_proc is {dock_proc.name} with PID {dock_proc.pid}")
    logger.info(f"train_proc is {train_proc.name} with PID {train_proc.pid}")

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
    still_running = False
    for proc in all_procs:
        if proc.exitcode != 0 and proc.exitcode is not None:
            error_on_exit = True
            logger.error(f"Process {proc.name} exited with code {proc.exitcode}")
            #raise Exception(f"Process {proc.name} exited with code {proc.exitcode}")
        elif proc.exitcode is None:
            still_running = True
            logger.warning(f"Process {proc.name} is still running unexpectedly")
        else:
            logger.info(f"Process {proc.name} exited successfully with code {proc.exitcode}")

    # If any process exited with an error code, raise an exception
    if error_on_exit:
        for proc in all_procs:
            proc.terminate()
        raise Exception("One or more processes exited with an error code")
    elif still_running:
        for proc in all_procs:
            proc.terminate()
        raise Exception("One or more processes unexpectedly still running")
    else:
        logger.info("All processes completed successfully")

    # If all proceesses completed successfully, join them
    train_proc.terminate()
    logger.info("Training process joined")
    dock_proc.terminate()
    logger.info("Docking simulation process joined")
    sorter_proc.terminate()
    logger.info("Sorting process joined")
    inf_proc.terminate()
    logger.info("Inference process joined")

    model_list_dd.restore(args.max_iter)
    logger.info(f"Saving top final simulated compounds ...")
    save_simulations(sim_dd, model_list_dd.checkpoint_id, number=top_candidate_number)

    # Close the dictionary
    logger.info("Closing the Dragon Dictionary and exiting ...")
    model_list_dd.destroy()
    data_dd.destroy()
    sim_dd.destroy()
    end_time = perf_counter()
    logger.info(f"Total time {end_time - start_time} seconds")
  