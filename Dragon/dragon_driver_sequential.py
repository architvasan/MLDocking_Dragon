import os
import sys
from time import perf_counter
import argparse
from typing import List
import random
import shutil
import pathlib
import logging
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
from driver_functions import max_data_dict_size, save_candidates, save_simulations, get_available_threads
from logging_config import driver_logger as logger

if __name__ == "__main__":

    # Import command line arguments
    parser = argparse.ArgumentParser(description='Distributed dictionary example')
    parser.add_argument('--inference_node_num', type=int, default=1,
                        help='number of nodes running inference')
    parser.add_argument('--sorting_node_num', type=int, default=1,
                        help='number of nodes running sorting')
    parser.add_argument('--simulation_node_num', type=int, default=1,
                        help='number of nodes running docking simulation')
    parser.add_argument('--training_node_num', type=int, default=1,
                        help='number of nodes running training')
    parser.add_argument('--managers_per_node', type=int, default=1,
                        help='number of managers per node for the dragon dict')
    parser.add_argument('--mem_per_node', type=int, default=8,
                        help='managed memory size per node for dictionary in GB')
    parser.add_argument('--mem_fraction', type=float, default=0.5,
                        help='Fraction of node memoty to be used for DDicts managed memory')
    parser.add_argument('--max_procs_per_node', type=int, default=10,
                        help='Maximum number of processes in a Pool')
    parser.add_argument('--max_iter', type=int, default=1,
                        help='Maximum number of iterations')
    parser.add_argument('--data_path', type=str, default="/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC-22-2D-smaller_files",
                        help='Path to pre-sorted SMILES strings to load')
    parser.add_argument('--load', type=str, default="False", choices=["False", "True"],
                        help='Perform data loading only')
    parser.add_argument('--inference_and_sort', type=str, default="False", choices=["False", "True"],
                        help='Perform loading, inference and sorting only')
    parser.add_argument('--sort', type=str, default="False", choices=["False", "True"],
                        help='Perform loading and sorting only')
    parser.add_argument('--num_files', type=int, default=128,
                        help='Number of files to load')                 
    args = parser.parse_args()

    # Start driver
    start_time = perf_counter()
    
    logger.info("Begun dragon driver")
    logger.info(f"Reading inference data from path: {args.data_path}")

    mp.set_start_method("dragon")

    # Get information about the allocation
    alloc = System()
    num_tot_nodes = int(alloc.nnodes)
    logger.info(f"Running on {num_tot_nodes} total nodes")
    tot_nodelist = alloc.nodes
    tot_mem = args.mem_per_node * num_tot_nodes

    # Get info about gpus and cpus
    gpu_devices = os.getenv("GPU_DEVICES")
    if gpu_devices is not None:
        gpu_devices = gpu_devices.split(",")
        num_gpus = len(gpu_devices)
    else:
        num_gpus = 0

    # Distribute nodes
    # TODO: Make node allocation more flexible
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


    # Set up and launch the dictionaries
    # There are 3 dictionaries:
    # 1. data dictionary for inference
    # 2. simulation dictionary for docking simulation results
    # 3. model and candidate dictionary for training
    # The model and candidate dictionary will be checkpointed
    
    # Get info on the number of files
    base_path = pathlib.Path(args.data_path)
    files, num_files = get_files(base_path)
    if num_tot_nodes <= 3:
        num_files = min(128, num_files)
    logger.info(f"There are {num_files} files")

    data_dict_mem, sim_dict_mem, model_list_dict_mem = max_data_dict_size(num_files, node_counts, max_pool_frac=args.mem_fraction)
    print(f"Setting data_dict size to {data_dict_mem} GB")
    print(f"Setting sim_dict size to {sim_dict_mem} GB")
    print(f"Setting model_list_dict size to {model_list_dict_mem} GB")

    # Check if total memory required exceeds available memory
    if data_dict_mem + sim_dict_mem + model_list_dict_mem > tot_mem:
        logger.info(f"Sum of dictionary sizes exceed total mem: {data_dict_mem=} {sim_dict_mem=} {model_list_dict_mem=} {tot_mem=}")
        raise Exception("Not enough memory for DDicts")

    # Convert memory sizes to bytes
    data_dict_mem *= (1024 * 1024 * 1024)
    sim_dict_mem *= (1024 * 1024 * 1024)
    model_list_dict_mem *= (1024 * 1024 * 1024)

    #TODO: Add node policies for dictionaries
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

    logger.debug("Here are the stats after data loading...")
    logger.debug("Data Dictionary stats:")
    logger.debug(data_dd.stats)

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
    
    logger.info("Here are the data DDict stats after data loading...")
    for stat in data_dd.stats:
        logger.info(f"manager_ID={stat.manager_id}, "+
                f"host_name={stat.hostname}, "+
                f"num_keys={stat.num_keys}, "+
                f"used_mem={stat.pool_utilization}, "+
                f"total_mem={stat.total_bytes}")
    
    if args.load == "True":
        sys.exit()

    # # Load pretrained model
    # load_pretrained_model(model_list_dd)
    # print("\nLoaded pretrained model",flush=True)

    # # Initialize simulated compounds list
    # if args.load == "False" and args.inference_and_sort == "False" and args.sort == "False":
    #     sim_dd.bput('simulated_compounds', [])
    
    # Number of top candidates to produce
    if num_tot_nodes <= 3:
        top_candidate_number = 1000
    else:
        top_candidate_number = 10000

    logger.info(f"Finished workflow setup in {(perf_counter()-start_time):.3f} seconds")

    # Start sequential loop
    max_iter = args.max_iter
    loop_iter = 0
    with open("driver_times.log", "a") as f:
        f.write(f"# iter  infer_time  sort_time  dock_time  train_time \n")
    while loop_iter < max_iter:
        logger.info(f"*** Start loop iter {loop_iter} ***")
        iter_start = perf_counter()
        logger.info(f"Current checkpoint: {model_list_dd.checkpoint_id}")

        # Launch the data inference component
        logger.info(f"Launching inference ...")
        inf_num_limit = None
        if num_tot_nodes < 3:
            inf_num_limit = 8
            logger.info(f"Running small test on {num_tot_nodes} nodes; limiting {inf_num_limit} keys per inference worker")
        num_proc_inference = num_gpus * node_counts["inference"]
        tic = perf_counter()
        inf_proc = mp.Process(
            target=launch_inference,
            args=(
                data_dd,
                model_list_dd,
                nodelists["inference"],
                num_proc_inference,
                inf_num_limit,
                ),
            )
        inf_proc.start()
        inf_proc.join()
        toc = perf_counter()
        infer_time = toc - tic
        logger.info(f"Executed inference mp.Process in {infer_time:.3f} seconds")
        if inf_proc.exitcode != 0:
            raise Exception("Inference failed!\n")
        
        # Launch data sorter component
        logger.info(f"Launching sorting ...")
        tic = perf_counter()
        # Add random compunds if desired
        #random_number = int(args.candidate_fraction*top_candidate_number) if loop_iter == 0 else 0
        random_number_fraction = 0
        sorting_threads = [get_available_threads(sequential_workflow=True)[0]]
        sorter_proc = mp.Process(target=sort_controller,
                                args=(
                                    data_dd,
                                    top_candidate_number,
                                    args.max_procs_per_node,
                                    nodelists['sorting'],
                                    sorting_threads,
                                    model_list_dd,
                                    random_number_fraction,
                                    ),
                                )
        sorter_proc.start()
        sorter_proc.join()
        if sorter_proc.exitcode != 0:
            raise Exception(f"Sorting failed {sorter_proc.exitcode=} \n")
        toc = perf_counter()
        sort_time = toc - tic
        logger.info(f"Executed sorting mp.Process with {num_keys} keys in {sort_time:.3f} seconds")
        if args.inference_and_sort == "True" or args.sort == "True":
            sys.exit()

        # Launch Docking Simulations
        logger.info(f"Launched docking simulations ...")
        tic = perf_counter()
        max_num_procs = min(args.max_procs_per_node, top_candidate_number//8)
        simulation_threads = get_available_threads(sequential_workflow=True)
        simulation_threads = simulation_threads[:max_num_procs]
        num_proc_simulation = len(simulation_threads)*node_counts["simulation"]

        dock_proc = mp.Process(
            target=launch_docking_sim,
            args=(sim_dd, 
                model_list_dd, 
                num_proc_simulation, 
                nodelists["simulation"],
                simulation_threads,),
        )
        dock_proc.start()
        dock_proc.join()
        if dock_proc.exitcode != 0:
            raise Exception("Docking sims failed\n")
        toc = perf_counter()
        dock_time = toc - tic
        logger.info(f"Executed docking mp.Process in {dock_time:.3f} seconds")

        # Launch Training
        logger.info(f"Launched Fine Tune Training...")
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
                ),
            )
        train_proc.start()
        train_proc.join()
        toc = perf_counter()
        train_time = toc - tic
        logger.info(f"Executed training mp.Process in {train_time} seconds")
        if train_proc.exitcode != 0:
            raise Exception("Training failed\n")
        iter_end = perf_counter()
        iter_time = iter_end - iter_start
        logger.info(f"Performed iter {loop_iter} in {iter_time} seconds")
        with open("driver_times.log", "a") as f:
            f.write(f"{loop_iter}  {infer_time}  {sort_time}  {dock_time}  {train_time}\n")
    
        model_list_dd.checkpoint()
        loop_iter += 1


    # Close the dictionary
    print("Closing the Dragon Dictionary and exiting ...", flush=True)
    model_list_dd.destroy()
    data_dd.destroy()
    sim_dd.destroy()
    end_time = perf_counter()
    print(f"Total time {end_time - start_time} seconds", flush=True)
