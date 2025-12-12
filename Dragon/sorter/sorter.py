from time import perf_counter
import argparse
import os
import random
import gc
import dragon
import logging
import multiprocessing as mp
from dragon.data.ddict.ddict import DDict
from dragon.native.process_group import ProcessGroup
from dragon.infrastructure.policy import Policy
from dragon.native.process import Process, ProcessTemplate, MSG_PIPE, MSG_DEVNULL
from dragon.infrastructure.connection import Connection
from dragon.native.machine import cpu_count, current, System, Node
from dragon.utils import host_id
from .sort_mpi import mpi_sort
import datetime

import time
import heapq
import socket
import traceback

import intel_extension_for_tensorflow as itex
from data_loader.data_loader_presorted import load_inference_data, initialize_worker
from logging_config import sort_logger as logger
from logging_config import driver_logger


MAX_BRANCHING_FACTOR = 5


def compare_candidate_results():
    """Compare candidate results from different sorting iterations."""
    # TODO: implement comparison logic to determine model improvement and convergence
    pass


def sort_controller(
    dd,
    top_candidate_number: int,
    max_procs: int,
    nodelist: list,
    thread_list: list,
    model_list_dd,
    random_number_fraction,
    max_iter=10,
    #continue_event=None,
    stop_event=None,
    new_model_event=None,
    barrier=None,
    checkpoint_interval_sec=60,
):

    sequential_workflow = stop_event is None
    
    logger.info("Starting Sort Controller")
    logger.info(f"Sorting for {top_candidate_number} candidates")

    # For asynchronous workflow, wait for checkpoint interval before starting
    if not sequential_workflow:
        time.sleep(checkpoint_interval_sec)

    continue_flag = True
    sorting_iter = 0
    while continue_flag:

        checkpoint_id = model_list_dd.checkpoint_id

        logger.info(f"Sorting on checkpoint {checkpoint_id} and iteration {sorting_iter}")

        #model_list_dd.sync_to_newest_checkpoint()
        #logger.info(f"Model list dictionary synced to newest checkpoint id {model_list_dd.checkpoint_id}")
        tic = perf_counter()

        random_number = int(random_number_fraction*top_candidate_number)
        logger.info(f"Adding {random_number} random candidates to training")
        if os.getenv("USE_MPI_SORT"):
            logger.info("Using MPI sort")
            max_sorter_procs = max_procs*len(nodelist)
            sorter_proc = mp.Process(target=distributed_mpi_sort, 
                                    args=(dd,
                                        top_candidate_number,
                                        max_sorter_procs, 
                                        nodelist,
                                        thread_list,
                                        model_list_dd,
                                        random_number,
                                        ),
                                    )
            sorter_proc.start()
            sorter_proc.join()
        else:
            logger.info("Using filter sort")
            sorter_proc = mp.Process(target=sort_dictionary,
                                    args=(
                                            dd,
                                            top_candidate_number,
                                            model_list_dd,
                                            random_number,
                                            ),
                                    )
            sorter_proc.start()
            sorter_proc.join()

        sorting_iter += 1
        # Make checkpoint decision
        if sequential_workflow:
            continue_flag = False
        else:
            if new_model_event.is_set():
                model_list_dd.checkpoint()
                checkpoint_id = model_list_dd.checkpoint_id
                driver_logger.info(f"Sort controller waiting at barrier on {sorting_iter=} and {checkpoint_id=}...")
                barrier.wait()
                logger.info(f"Sort controller advanced to checkpoint {model_list_dd.checkpoint_id}")
                
                # Check whether to continue async workflow
                # if checkpoint_id > max_iter: 
                #     continue_event.clear()
                #     driver_logger.info(f"Clearing continue_event after {max_iter} checkpoints for testing purposes")
                continue_flag = not stop_event.is_set()

                # For asynchronous workflow, wait for checkpoint interval before sorting with new inference results
                if not sequential_workflow and continue_flag:
                    time.sleep(checkpoint_interval_sec)
        

def get_largest(dd, out_queue, num_return_sorted, checkpoint_id):
    # get num_return_sorted values from the manager
    # reflected in dd (i.e. dd is a manager directed
    # subset of a ddict).
    try:
        tic = perf_counter()
        keys = dd.keys()
        #keys = [k for k in keys if "model" not in k and "iter" not in k]
        this_value = []

        for key in keys:
            if "model" not in key and "iter" not in key:
                val = dd[key]
                # Skip values from models before checkpoint_id
                if val['model_iter'] < checkpoint_id:
                    continue
                num_smiles = len(val['inf'])
                this_value.extend(zip(val["inf"], 
                                        val["smiles"], 
                                        [val['model_iter'] for _ in range(num_smiles)]))
                this_value = heapq.nlargest(
                    num_return_sorted, this_value, key=lambda x: x[0]
                )

        # If EOFError is raised, the receiving side closed the queue
        try:
            for i in range(num_return_sorted):
                out_queue.put(this_value[i])
        except EOFError:
            pass
        except IndexError:
            pass

        #print(perf_counter()-tic,flush=True)

    except Exception as ex:
        tb = traceback.format_exc()
        print(
            "There was an exception in get_largest: %s\n Traceback: %s" % (ex, tb),
            flush=True,
        )


def comparator(x, y):
    return x[0] > y[0]


def sort_dictionary(dd: DDict, num_return_sorted, cdd: DDict, random_number: int):
    
    tic_start = perf_counter()
    logger.info(f"Finding the best {num_return_sorted} candidates.")
    checkpoint_id = cdd.checkpoint_id
    candidate_list = []

    tic_filter = perf_counter()
    with dd.filter(get_largest, (num_return_sorted, checkpoint_id,), comparator, branching_factor=4) as candidates:
        for candidate in candidates:
            candidate_list.append(candidate)
            if len(candidate_list) == num_return_sorted:
                break
    toc_filter = perf_counter()

    if len(candidate_list) == 0:
        logger.info(f"No candidates found for sorting")
        return

    logger.info("HERE IS THE CANDIDATE LIST (first 10 only)")
    logger.info("******************************************")
    logger.info(candidate_list[:10])

    candidate_inf,candidate_smiles,candidate_model_iter = zip(*candidate_list)
    non_zero_infs = len([cinf for cinf in candidate_inf if cinf != 0])
    sort_val = {"inf": list(candidate_inf), 
                "smiles": list(candidate_smiles), 
                "model_iter": list(candidate_model_iter)}

    current_sort_iter = cdd.checkpoint_id
    
    tic_w = perf_counter()
    cdd.bput("current_sort_list", sort_val)
    toc_w = perf_counter()
    toc_end = perf_counter()

    logger.info(f"Finished filter sort")

    io_time =(toc_w-tic_w)
    filter_time = toc_filter - tic_filter

    logger.info(f"Performed sorting of {num_return_sorted} compounds: total={toc_end-tic_start}, filter={filter_time}, IO={io_time}")

    if random_number > 0:
        print(f"Adding {random_number} random candidates to training", flush=True)
        num_procs = min(len(nodelist), random_number)
        random_num_per_proc = max(int(random_number/num_procs), 1)
        print(f"Starting a pool with {num_procs} processes each adding {random_num_per_proc} random candidates", flush=True)
        pool = mp.Pool(num_procs, 
                    initializer=initialize_worker, 
                    initargs=(dd,), 
                    )
        out = pool.imap_unordered(make_random_compound_selection, 
                                [random_num_per_proc for _ in range(num_procs)])

        random_smiles = []
        random_inf = []
        random_model = []
        for result in out:
            for r in result:
                sm,sc,mi = r
                random_smiles.append(sm)
                random_inf.append(sc)
                random_model.append(mi)
        pool.close()
        pool.join()
        print(f"Randomly sampled {len(random_smiles)} random smiles for simulation", flush=True)
        cdd['random_compound_sample'] = {'smiles': random_smiles,
                                        'inf': random_inf,
                                        'model_iter': random_model,}
    

def make_random_compound_selection(random_number):
    try:
        
        me = mp.current_process()
        dd = me.stash["ddict"]

        #alloc = System()
        #num_tot_nodes = int(alloc.nnodes) 
        #num_random_per_node = max(int(random_number/num_tot_nodes), 1)
        num_random_per_node = random_number


        random_selection = []

        # Select num_random_per_node random keys
        #current_host = host_id()
        #manager_nodes = dd.manager_nodes
        #key_list = []
        #for i in range(len(manager_nodes)):
        #    if manager_nodes[i].h_uid == current_host:
        #        dm = dd.manager(i)
        #        key_list.extend(dm.keys())
        #        # Filter out keys containing model or iter info
        key_list = dd.local_keys()
        key_list = [key for key in key_list if "model" not in key and "iter" not in key]

        if len(key_list) > 0:
            irand = [random.randint(0, len(key_list)-1) for _ in range(num_random_per_node)]
            for i,k in enumerate(key_list):
                frequency = irand.count(i)
                if frequency > 0:
                    val = dd[k]
                    smiles = val['smiles']
                    inf_val = val['inf']
                    model_iter = val['model_iter']
                    for f in range(frequency):
                        jrand = random.randint(0,len(smiles)-1)
                        random_selection.append((smiles[jrand],inf_val[jrand], model_iter))
    except Exception as e:
        logger.error(f"Pool worker failed with this error {e}")
        raise Exception(e)
    
    return random_selection


def distributed_mpi_sort(dd: DDict, 
                       num_return_sorted: int, 
                       num_procs: int, 
                       nodelist, cdd: DDict,
                       thread_list: list,
                       random_number):
   
    max_num_procs_pn = num_procs//len(nodelist)
    run_dir = os.getcwd()
    key_list = dd.keys()
    key_list = [key for key in key_list if "iter" not in key and "model" not in key]
    
    num_keys = len(key_list)

    keys_per_node = num_keys//len(nodelist)
    min_direct_sort_num = 4

    direct_sort_num = max(num_keys//num_procs+1,min_direct_sort_num)
    num_procs_pn = keys_per_node // direct_sort_num

    spacing = len(thread_list)//num_procs_pn
    thread_list = thread_list[::spacing][:num_procs_pn]
    
    logger.info(f"Direct sorting {direct_sort_num} keys per process")

    global_policy = Policy(distribution=Policy.Distribution.BLOCK)
    grp = ProcessGroup(policy=global_policy, pmi_enabled=True)

    logger.info(f"Launching sorting process group {nodelist}")
    for node in nodelist:
        node_name = Node(node).hostname
        local_policy = Policy(placement=Policy.Placement.HOST_NAME, 
                            host_name=node_name, 
                            cpu_affinity=thread_list,)
        grp.add_process(nproc=num_procs_pn, 
                            template=ProcessTemplate(target=mpi_sort, 
                                                    args=(dd, num_keys, num_return_sorted,cdd), 
                                                    policy=local_policy,
                                                    cwd=run_dir))
    logger.info(f"Added processes to sorting group")
    grp.init()
    grp.start()
    logger.info(f"Starting Process Group for Sorting")
    grp.join()
    logger.info(f"Process Group for Sorting has joined")
    grp.close()

    logger.info("Getting random compounds")
    # Grab random compounds from each node

    if random_number > 0:
        alloc = System()
        num_nodes = min(int(alloc.nnodes), random_number)
        pool = mp.Pool(num_nodes, 
                    initializer=initialize_worker, 
                    initargs=(dd,), 
                    )
        out = pool.imap_unordered(make_random_compound_selection, 
                                [random_number for _ in range(num_nodes)])

        random_smiles = []
        random_inf = []
        random_model = []
        for result in out:
            for r in result:
                sm,sc,mi = r
                random_smiles.append(sm)
                random_inf.append(sc)
                random_model.append(mi)
        pool.close()
        pool.join()
        logger.info(f"Randomly sampled {len(random_smiles)} random smiles for simulation")
        cdd['random_compound_sample'] = {'smiles': random_smiles,
                                        'inf': random_inf,
                                        'model_iter': random_model,}
   

def create_dummy_data(_dict,num_managers):

    NUMKEYS = 100

    for i in range(NUMKEYS):
        key = f"{i%num_managers}_{i}"
        _dict[key] = {
            "inf": [random.randrange(0, 1000, 1) for j in range(10)],
            "smiles": [
                "".join(
                    [
                        random.choice(string.ascii_uppercase + string.digits)
                        for k in range(20)
                    ]
                )
                for j in range(10)
            ],
        }


if __name__ == "__main__":
    # Import command line arguments

    parser = argparse.ArgumentParser(description='Distributed dictionary example')
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='number of nodes the dictionary distributed across')
    parser.add_argument('--managers_per_node', type=int, default=1,
                        help='number of managers per node for the dragon dict')
    parser.add_argument('--mem_per_node', type=int, default=1,
                        help='managed memory size per node for dictionary in GB')
    parser.add_argument('--max_procs_per_node', type=int, default=32,
                        help='Maximum number of processes in a Pool')
    parser.add_argument('--data_path', type=str, default="/eagle/hpe_dragon_collab/csimpson/ZINC-22-presorted/tiny",
                        help='Path to pre-sorted SMILES strings to load')

    args = parser.parse_args()

    
    mp.set_start_method("dragon")

    total_mem_size = args.mem_per_node * args.num_nodes * (1024 * 1024 * 1024)
    dd = DDict(args.managers_per_node, args.num_nodes, total_mem_size)
    print("Launched Dragon Dictionary \n", flush=True)

    loader_proc = mp.Process(
        target=create_dummy_data,
        args=(dd, args.num_nodes * args.managers_per_node),
    )  # ignore_exit_on_error=True)
    loader_proc.start()
    print("Process started", flush=True)


    # Get information about the allocation
    alloc = System()
    num_tot_nodes = int(alloc.nnodes)
    tot_nodelist = alloc.nodes

    # Start distributed dictionary and load data
    tot_mem = num_tot_nodes*args.mem_per_node
    data_dictionary_mem_fraction = 0.1

    data_dict_mem = max(int(tot_mem), num_tot_nodes)
    candidate_dict_mem = max(int(tot_mem*(1.-data_dictionary_mem_fraction)), num_tot_nodes)
    print(f"Setting data_dict size to {data_dict_mem} GB and candidate_dict size to {candidate_dict_mem} GB")
    data_dict_mem *= (1024*1024*1024)
    candidate_dict_mem *= (1024*1024*1024)

    data_dd = DDict(args.managers_per_node, num_tot_nodes, data_dict_mem, trace=True)
    print(f"Launched Dragon Dictionary for inference with total memory size {data_dict_mem}", flush=True)
    print(f"on {num_tot_nodes} nodes", flush=True)
    print(f"{data_dd.stats=}")

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
        ),
    )
    loader_proc.start()
    loader_proc.join()

    print("Process ended", flush=True)

    print(f"Number of keys in dictionary is {len(dd.keys())}", flush=True)

    candidate_queue = mp.Queue()
    top_candidate_number = 10
    sorter_proc = mp.Process(
        target=sort_dictionary,
        args=(
            dd,
            top_candidate_number,
            args.max_procs_per_node * args.num_nodes,
            dd.keys(),
            candidate_queue,
        ),
    )  # ignore_exit_on_error=True)
    sorter_proc.start()
    print("Process started", flush=True)

    sorter_proc.join()
    print("Process ended", flush=True)
    top_candidates = candidate_queue.get(timeout=None)
    print(f"{top_candidates=}")

    # # Launch the data loader
    # print("Loading inference data into Dragon Dictionary ...", flush=True)
    # tic = perf_counter()
    # load_inference_data(dd, args.data_path, args.max_procs)
    # toc = perf_counter()
    # load_time = toc - tic
    # print(f"Loaded inference data in {load_time:.3f} seconds \n", flush=True)

    # Close the dictionary
    print("Done, closing the Dragon Dictionary", flush=True)
    dd.destroy()

