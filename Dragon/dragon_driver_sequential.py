import os
import sys
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

#import logging
#log = logging.getLogger(__name__)
#logging.basicConfig(
#    level=logging.INFO,
#    format='%(levelname)s - %(message)s'
#)

from data_loader.data_loader_presorted import load_inference_data
from inference.launch_inference import launch_inference
from sorter.sorter import sort_controller
from docking_sim.launch_docking_sim import launch_docking_sim
from training.launch_training import launch_training

#from sorter.sorter import sort_dictionary_pg, sort_dictionary
#from docking_sim.launch_docking_sim import launch_docking_sim
#from training.launch_training import launch_training
from data_loader.data_loader_presorted import get_files
from data_loader.model_loader import load_pretrained_model
from driver_functions import max_data_dict_size, save_candidates, save_simulations


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
    parser.add_argument('--dictionary_timeout', type=int, default=10,
                        help='Timeout for Dictionary in seconds')
    parser.add_argument('--data_path', type=str, default="/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC-22-2D-smaller_files",
                        help='Path to pre-sorted SMILES strings to load')
    parser.add_argument('--logging', type=str, default="info", choices=["info","debug"],
                        help='Logging level')
    parser.add_argument('--load', type=str, default="False", choices=["False", "True"],
                        help='Perform data loading only')
    parser.add_argument('--inference_and_sort', type=str, default="False", choices=["False", "True"],
                        help='Perform loading, inference and sorting only')
    parser.add_argument('--sort', type=str, default="False", choices=["False", "True"],
                        help='Perform loading and sorting only')
    parser.add_argument('--pool_chunksize', type=int, default=1,
                        help='Chunksize to use for the data loader mp.Pool')
    parser.add_argument('--candidate_fraction', type=float, default=0.0,
                        help='Fraction of the compund list to inclide for training')
    args = parser.parse_args()

    # Start driver
    start_time = perf_counter()
    print("Begun dragon driver")
    print(f"Reading inference data from path: {args.data_path}", flush=True)
    mp.set_start_method("dragon")

    if args.sort == "True":
        os.environ['TEST_SORTING'] = 'True'

    # Get information about the allocation
    alloc = System()
    num_tot_nodes = int(alloc.nnodes)
    print(f"Running on {num_tot_nodes} total nodes",flush=True)
    tot_nodelist = alloc.nodes
    tot_mem = args.mem_per_node * num_tot_nodes

    # Distribute nodes
    print()
    if args.inference_node_num != args.sorting_node_num:
        raise("Inference and sorting nodes must be the same for this colocated deployment!")
    if num_tot_nodes != (args.inference_node_num + args.simulation_node_num): 
        raise(f"Node partitioning not valid! Inference and simulation nodes must add up to total nodes.")
    node_counts = {
        "inference": args.inference_node_num,
        "sorting": args.sorting_node_num,
        "simulation": args.simulation_node_num,
        "training": args.training_node_num, 
    }
    nodelist = {
        "inference": tot_nodelist[:args.inference_node_num],
        "sorting": tot_nodelist[:args.sorting_node_num],
        "simulation": tot_nodelist[args.inference_node_num:],
        "training": [tot_nodelist[-1]]
    }
    for key, val in nodelist.items():
        print(f"Component {key} running on {[Node(node).hostname for node in val]} nodes",flush=True)

    # Get info about gpus and cpus
    gpu_devices = os.getenv("GPU_DEVICES")
    if gpu_devices is not None:
        gpu_devices = gpu_devices.split(",")
        num_gpus = len(gpu_devices)
    else:
        num_gpus = 0

    # Set up and launch the dictionaries
    # There are 3 dictionaries:
    # 1. data dictionary for inference
    # 2. simulation dictionary for docking simulation results
    # 3. model and candidate dictionary for training
    # The model and candidate dictionary will be checkpointed
    if num_tot_nodes <= 3: 
        num_files = 24
    else:
        base_path = pathlib.Path(args.data_path)
        files, num_files = get_files(base_path)
    data_dict_mem, sim_dict_mem, model_list_dict_mem = max_data_dict_size(num_files, node_counts, max_pool_frac=args.mem_fraction)
    print(f"Setting data_dict size to {data_dict_mem} GB")
    print(f"Setting sim_dict size to {sim_dict_mem} GB")
    print(f"Setting model_list_dict size to {model_list_dict_mem} GB")

    # Convert memory sizes to bytes
    data_dict_mem *= (1024 * 1024 * 1024)
    sim_dict_mem *= (1024 * 1024 * 1024)
    model_list_dict_mem *= (1024 * 1024 * 1024)

    # Initialize Dragon Dictionaries for inference, docking simulation, and model 
    
    #data_dd = DDict(args.managers_per_node, num_tot_nodes, data_dict_mem)
    #print(f"Launched Dragon Dictionary for inference with total memory size {data_dict_mem} on {num_tot_nodes} nodes", flush=True)
    #sim_dd = DDict(args.managers_per_node, num_tot_nodes, sim_dict_mem)
    #print(f"Launched Dragon Dictionary for docking simulation with total memory size {sim_dict_mem} on {num_tot_nodes} nodes", flush=True)
    #model_list_dd = DDict(args.managers_per_node, num_tot_nodes, model_list_dict_mem, working_set_size=10, wait_for_keys=True)
    #print(f"Launched Dragon Dictionary for model list with total memory size {model_list_dict_mem} on {num_tot_nodes} nodes", flush=True)
    
    data_dd_cpu_bind = os.getenv("DATA_DD_CPU_AFFINITY").split(",")
    data_dd_policy = [Policy(placement=Policy.Placement.HOST_NAME, 
                             host_name=Node(nodelist["inference"][node]).hostname,
                             cpu_affinity=data_dd_cpu_bind) \
                      for node in range(len(nodelist["inference"]))]
    tic = perf_counter()
    data_dd = DDict(None, 
                    None, 
                    data_dict_mem, 
                    policy=data_dd_policy,
                    managers_per_policy=args.managers_per_node)
    toc = perf_counter()
    print(f"Launched Dragon Dictionary for inference with total memory size {data_dict_mem} on {node_counts['inference']} nodes in {toc-tic} seconds", flush=True)
    
    if args.load == "False" and args.inference_and_sort == "False" and args.sort == "False":
        sim_dd_cpu_bind = os.getenv("SIM_DD_CPU_AFFINITY").split(",")
        sim_dd_policy = [Policy(placement=Policy.Placement.HOST_NAME, 
                                host_name=Node(nodelist["simulation"][node]).hostname,
                                cpu_affinity=sim_dd_cpu_bind) \
                        for node in range(len(nodelist["simulation"]))]
        tic = perf_counter()
        sim_dd = DDict(None, 
                    None, 
                    sim_dict_mem,
                    policy=sim_dd_policy,
                    managers_per_policy=args.managers_per_node)
        toc = perf_counter()
        print(f"Launched Dragon Dictionary for docking simulation with total memory size {sim_dict_mem} on {node_counts['simulation']} nodes in {toc-tic} seconds", flush=True)

    model_dd_cpu_bind = os.getenv("MODEL_DD_CPU_AFFINITY").split(",")
    model_dd_policy = Policy(cpu_affinity=model_dd_cpu_bind)
    tic = perf_counter()
    model_list_dd = DDict(args.managers_per_node, 
                          num_tot_nodes, 
                          model_list_dict_mem, 
                          policy=model_dd_policy, 
                          working_set_size=10)
    toc = perf_counter()
    print(f"Launched Dragon Dictionary for model list with total memory size {model_list_dict_mem} on {num_tot_nodes} nodes in {toc-tic} seconds", flush=True)

    # Load data into the data dictionary
    print("\nLoading inference data into Dragon Dictionary ...", flush=True)
    if num_tot_nodes <= 3: 
        num_procs = num_files
    else:
        files_per_proc = 1000
        max_avail_cores = (args.max_procs_per_node-len(data_dd_cpu_bind)-len(model_dd_cpu_bind)) * node_counts["inference"]
        num_procs = min(num_files, int(num_files / files_per_proc))
        num_procs = min(num_procs, max_avail_cores)
    tic = perf_counter()
    loader_proc = mp.Process(
        target=load_inference_data,
        args=(
            data_dd,
            args.data_path,
            num_procs,
            node_counts["inference"] * args.managers_per_node,
        ),
        kwargs={
            'num_files': num_files,
            'nodelist': None,
            'load_split_factor': 1,
            'chunksize': args.pool_chunksize,
        }
    )
    loader_proc.start()
    loader_proc.join()

    print("Here are the stats after data loading...")
    print("Data Dictionary stats:", flush=True)
    print(data_dd.stats)

    # Load pretrained model
    load_pretrained_model(model_list_dd)

    # Initialize simulated compounds list
    model_list_dd.bput('simulated_compounds', [])

    # Report Loading time
    toc = perf_counter()
    load_time = toc - tic
    if loader_proc.exitcode == 0:
        print(f"Executed loader mp.Process in {load_time:.3f} seconds", flush=True)
    else:
        raise Exception(f"Data loading failed with exception {loader_proc.exitcode}")

    if args.logging == "debug":
        print("\nHere are the data DDict stats after data loading...",flush=True)
        for stat in data_dd.stats:
            print(f"manager_ID={stat.manager_id}, ",
                  f"host_name={stat.hostname}, ",
                  f"num_keys={stat.num_keys}, ",
                  f"used_mem={stat.pool_utilization}, ",
                  f"total_mem={stat.total_bytes}",
                  flush=True)
    
    if args.load == "True":
        sys.exit()

    # Load pretrained model
    load_pretrained_model(model_list_dd)
    print("\nLoaded pretrained model",flush=True)

    # Initialize simulated compounds list
    if args.load == "False" and args.inference_and_sort == "False" and args.sort == "False":
        sim_dd.bput('simulated_compounds', [])

    # Update driver log
    with open("driver_times.log", "w") as f:
        f.write(f"# {load_time=}\n")
    num_keys = len(data_dd.keys())
    with open("driver_times.log", "a") as f:
        f.write(f"# {num_keys=}\n")
    with open("driver_times.log", "a") as f:
        f.write(f"# {num_files=}\n")
    
    # Number of top candidates to produce
    if num_tot_nodes <= 3:
        top_candidate_number = 100
    else:
        top_candidate_number = 10000

    print(f"\nFinished workflow setup in {(perf_counter()-start_time):.3f} seconds\n", flush=True)

    # Start sequential loop
    max_iter = args.max_iter
    iter = 0
    with open("driver_times.log", "a") as f:
        f.write(f"# iter  infer_time  sort_time  dock_time  train_time \n")
    while iter < max_iter:
        print(f"\n*** Start loop iter {iter} ***")
        iter_start = perf_counter()

        print(f"Current checkpoint: {model_list_dd.checkpoint_id}", flush=True)

        # Launch the data inference component
        if args.sort == "False":
            print(f"Launching inference ...", flush=True)
            inf_num_limit = None
            if num_tot_nodes == 1:
                inf_num_limit = 8
                print(f"Running small test on {num_tot_nodes}; limiting {inf_num_limit} keys per inference worker")

            tic = perf_counter()
            inf_proc = mp.Process(
                target=launch_inference,
                args=(
                    data_dd,
                    model_list_dd,
                    iter,
                    nodelist["inference"],
                ),
                kwargs={
                'inf_num_limit': inf_num_limit,
                }
            )
            inf_proc.start()
            inf_proc.join()
            toc = perf_counter()
            infer_time = toc - tic
            print(f"Executed inference mp.Process in {infer_time:.3f} seconds \n", flush=True)
            if inf_proc.exitcode != 0:
                raise Exception("Inference failed!\n")
        
        # Launch data sorter component
        print(f"Launching sorting ...", flush=True)
        tic = perf_counter()
        if iter == 0:
            model_list_dd.bput("max_sort_iter",-1)
            model_list_dd.bput('current_sort_iter', -1)
        
        # Add random compunds if desired
        random_number = int(args.candidate_fraction*top_candidate_number) if iter == 0 else 0
        if os.getenv("USE_MPI_SORT"):
            print("Using MPI sort",flush=True)
            max_sorter_procs = (args.max_procs_per_node-len(data_dd_cpu_bind)-len(model_dd_cpu_bind)) * node_counts["sorting"]
            sorter_proc = mp.Process(target=sort_dictionary_pg, 
                                     args=(data_dd,
                                           top_candidate_number,
                                           max_sorter_procs, 
                                           nodelist["sorting"],
                                           model_list_dd,
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
                                            nodelist["sorting"],
                                            model_list_dd,
                                            random_number,
                                            ),
                                      )
            sorter_proc.start()
            sorter_proc.join()
        if sorter_proc.exitcode != 0:
            raise Exception("Sorting failed\n")
        toc = perf_counter()
        sort_time = toc - tic
        print(f"Executed sorting mp.Process with {num_keys} keys in {sort_time:.3f} seconds \n", flush=True)
        if args.inference_and_sort == "True" or args.sort == "True":
            sys.exit()

        # Launch Docking Simulations
        print(f"Launched docking simulations ...", flush=True)
        tic = perf_counter()
        max_num_procs = top_candidate_number//1
        dock_proc = mp.Process(
            target=launch_docking_sim,
            args=(model_list_dd, 
                  sim_dd, 
                  iter, 
                  max_num_procs, 
                  nodelist["simulation"]),
        )
        dock_proc.start()
        dock_proc.join()
        if dock_proc.exitcode != 0:
            raise Exception("Docking sims failed\n")
        toc = perf_counter()
        dock_time = toc - tic
        print(f"Executed docking mp.Process in {dock_time:.3f} seconds \n", flush=True)

        # Launch Training
        print(f"Launched Fine Tune Training", flush=True)
        tic = perf_counter()
        BATCH = 64
        EPOCH = 150
        train_proc = mp.Process(
            target=launch_training,
            args=(
                model_list_dd,
                sim_dd,
                iter,
                nodelist["training"][0],  # training is always 1 node
                BATCH,
                EPOCH,
            ),
        )
        train_proc.start()
        train_proc.join()
        toc = perf_counter()
        train_time = toc - tic
        print(f"Executed training mp.Process in {train_time} seconds \n", flush=True)
        if train_proc.exitcode != 0:
            raise Exception("Training failed\n")
        iter_end = perf_counter()
        iter_time = iter_end - iter_start
        print(
            f"Performed iter {iter} in {iter_time} seconds \n", flush=True
        )
        with open("driver_times.log", "a") as f:
            f.write(f"{iter}  {infer_time}  {sort_time}  {dock_time}  {train_time}\n")

        if inf_proc.exitcode != 0:
            raise Exception("Inference failed!\n")
    

        # Launch data sorter component
        tic = perf_counter()
        print(f"Launching sorting ...", flush=True)
        random_number_fraction = 0.0

        sorter_proc = mp.Process(target=sort_controller,
                                    args=(
                                        data_dd,
                                        top_candidate_number,
                                        args.max_procs_per_node,
                                        nodelists['sorting'],
                                        model_list_dd,
                                        random_number_fraction,
                                        ),
                                    )

        sorter_proc.start()
        sorter_proc.join()
        if sorter_proc.exitcode != 0:
            raise Exception("Sorting failed!\n")

        toc = perf_counter()
        sort_time = toc - tic
        print(f"Performed sorting of {num_keys} keys in {sort_time:.3f} seconds \n", flush=True)


        # Launch Docking Simulations
        print(f"Launched Docking Simulations", flush=True)
        tic = perf_counter()

        num_procs = args.max_procs_per_node * node_counts["docking"]
        num_procs = min(num_procs, top_candidate_number//4)
        num_procs = max(num_procs, node_counts["docking"])
        dock_proc = mp.Process(
            target=launch_docking_sim,
            args=(sim_dd, 
                    model_list_dd, 
                    num_procs, 
                    nodelists["docking"],),
        )

        dock_proc.start()
        dock_proc.join()
        toc = perf_counter()
        dock_time = toc - tic
        #os.rename("finished_run_docking.log", f"finished_run_docking_{iter}.log")
        print(f"Performed docking in {dock_time:.3f} seconds \n", flush=True)
        
        print(f"Candidate Dictionary stats:", flush=True)
        print(sim_dd.stats)
        if dock_proc.exitcode != 0:
            raise Exception("Docking sims failed\n")


        # Checkpoint the model_list_dd
        model_list_dd.checkpoint()

    #     # Launch Training
        print(f"Launched Fine Tune Training", flush=True)
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

    
        iter += 1


    # Close the dictionary
    print("Closing the Dragon Dictionary and exiting ...", flush=True)
    model_list_dd.destroy()
    data_dd.destroy()
    sim_dd.destroy()
    end_time = perf_counter()
    print(f"Total time {end_time - start_time} seconds", flush=True)
