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

from data_loader.data_loader_presorted import load_inference_data, initialize_worker
from logging_config import sort_logger as logger


MAX_BRANCHING_FACTOR = 5

global data_dict
data_dict = None


def init_worker(q):
    global data_dict
    data_dict = q.get()
    return


def filter_candidate_keys(ckeys: list):
    ckeys = [
        key for key in ckeys if "iter" not in key and key[0] != "d" and key[0] != "l"
    ]
    return ckeys


def compare_candidate_results(
    candidate_dict,
    continue_event,
    num_return_sorted,
    ncompare=3,
    max_iter=100,
    purge=True,
):

    print(f"Comparing Candidate Lists")
    end_workflow = False
    candidate_keys = candidate_dict.keys()
    sort_iter = 0
    if "sort_iter" in candidate_keys:
        sort_iter = candidate_dict["sort_iter"]
        # candidate_keys.remove('iter')
    candidate_keys = filter_candidate_keys(candidate_keys)
    print(f"{candidate_keys=}")
    # ncompare = min(ncompare,len(candidate_keys))
    candidate_keys.sort(reverse=True)
    # print(f"{candidate_keys=}")
    num_top_candidates = 0
    if len(candidate_keys) > 0:
        num_top_candidates = len(candidate_dict[candidate_keys[0]]["smiles"])

    if sort_iter > max_iter:
        # end if maximum number of iterations reached
        end_workflow = True
        print(
            f"Ending workflow: sort_iter {sort_iter} exceeded max_iter {max_iter}",
            flush=True,
        )
    elif len(candidate_keys) > ncompare and num_top_candidates == num_return_sorted:
        # look for unique entries in most recent ncompare lists
        # only do this if there are enough lists to compare and if enough candidates have been identified

        not_in_common = []
        model_iters = []
        print(f"{candidate_dict=}", flush=True)
        for i in range(ncompare):
            for j in range(ncompare):
                if i < j:
                    ckey_i = candidate_keys[i]
                    ckey_j = candidate_keys[j]
                    not_in_common += list(
                        set(candidate_dict[ckey_i]["smiles"])
                        ^ set(candidate_dict[ckey_j]["smiles"])
                    )
                    model_iter_i = list(set(candidate_dict[ckey_i]["model_iter"]))
                    model_iter_j = list(set(candidate_dict[ckey_j]["model_iter"]))
                    model_iters += model_iter_i
                    model_iters += model_iter_j
        print(f"Number not in common {len(not_in_common)}")
        # ncompare consecutive lists are identical, end workflow
        print(f"{model_iters=}")

        # End workflow if ncompare lists with unique model_iters are idententical
        if len(not_in_common) == 0 and len(model_iters) == 2 * ncompare:
            if len(set(model_iters)) == ncompare:
                print(f"Ending workflow: {ncompare} lists identical", flush=True)
                end_workflow = True

        # If purge, only keep ncompare sorting lists
        if purge:
            if len(candidate_keys) > ncompare:
                del candidate_dict[candidate_keys[-1]]

    print(f"{end_workflow=}")
    if end_workflow:
        continue_event.clear()


def save_top_candidates_list(candidate_dict):
    ckeys = filter_candidate_keys(candidate_dict.keys())
    if len(ckeys) > 0:
        max_ckey = max(ckeys)
        top_candidates = candidate_dict[max_ckey]
        top_smiles = top_candidates["smiles"]
        lines = [sm + "\n" for sm in top_smiles]

        with open(f"top_candidates.out", "w") as f:
            f.writelines(lines)


def sort_controller(
    dd,
    top_candidate_number: int,
    max_procs: int,
    nodelist: list,
    model_list_dd,
    random_number_fraction,
    continue_event=None,
    checkpoint_interval_min=2,
):

    iter = 0
    
    logger.info("Starting Sort Controller")
    logger.info(f"Sorting for {top_candidate_number} candidates")

    ckeys = model_list_dd.keys()
    if "current_sort_iter" not in ckeys:
        model_list_dd.bput("current_sort_iter", -1)

    continue_flag = True

    while continue_flag:

        # Wait for sort interval
        if continue_event is not None:
            time.sleep(checkpoint_interval_min * 60)

        logger.info(f"Starting iter {iter}")
        tic = perf_counter()

        random_number = int(random_number_fraction*top_candidate_number)
        logger.info(f"Adding {random_number} random candidates to training")
        if os.getenv("USE_MPI_SORT"):
            logger.info("Using MPI sort")
            max_sorter_procs = max_procs*len(nodelist)
            sorter_proc = mp.Process(target=sort_dictionary_pg, 
                                    args=(dd,
                                        top_candidate_number,
                                        max_sorter_procs, 
                                        nodelist,
                                        model_list_dd,
                                        random_number,
                                        ),
                                    )
            sorter_proc.start()
            sorter_proc.join()
        else:
            logger.info("Using filter sort")
    #TODO: pass in checkpoint id to sorting
            sorter_proc = mp.Process(target=sort_dictionary,
                                    args=(
                                            dd,
                                            top_candidate_number,
                                            model_list_dd,
                                            ),
                                    )
            sorter_proc.start()
            sorter_proc.join()
        # if sorter_proc.exitcode != 0:
        #     raise Exception("Sorting failed\n")

        if continue_event is not None:
            if iter > 10: continue_event.clear()
            continue_flag = continue_event.is_set()
        else:
            continue_flag = False
    


#TODO: pass in checkpoint_id
def get_largest(dd, out_queue, num_return_sorted):
    # get num_return_sorted values from the manager
    # reflected in dd (i.e. dd is a manager directed
    # subset of a ddict).
    try:
        keys = dd.keys()
        keys = [k for k in keys if "model" not in k and "iter" not in k]
        this_value = []

        for key in keys:
            val = dd[key]
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

    except Exception as ex:
        tb = traceback.format_exc()
        print(
            "There was an exception in get_largest: %s\n Traceback: %s" % (ex, tb),
            flush=True,
        )


def comparator(x, y):
    return x[0] > y[0]


def sort_dictionary(dd: DDict, num_return_sorted, cdd: DDict):

    logger.info(f"Finding the best {num_return_sorted} candidates.")
    candidate_list = []

    with dd.filter(get_largest, (num_return_sorted,), comparator) as candidates:
        for candidate in candidates:
            candidate_list.append(candidate)
            if len(candidate_list) == num_return_sorted:
                break

    logger.info("HERE IS THE CANDIDATE LIST (first 10 only)")
    logger.info("******************************************")
    logger.info(candidate_list[:10])

    candidate_inf,candidate_smiles,candidate_model_iter = zip(*candidate_list)
    non_zero_infs = len([cinf for cinf in candidate_inf if cinf != 0])
    sort_val = {"inf": list(candidate_inf), 
                "smiles": list(candidate_smiles), 
                "model_iter": list(candidate_model_iter)}

    current_sort_iter = cdd.bget("current_sort_iter")
    if current_sort_iter > -1:
        current_sort_list = cdd.bget("current_sort_list")
        cdd[str(current_sort_iter)] = current_sort_list 

    new_sort_iter = int(current_sort_iter + 1)
    cdd.bput("current_sort_iter", new_sort_iter)
    cdd.bput("current_sort_list", sort_val)

    logger.info(f"Finished filter sort")
    #cdd[ckey] = sort_val
    #cdd["sort_iter"] = int(ckey)
    #cdd["max_sort_iter"] = ckey
    

def make_random_compound_selection(random_number):
    try:
        
        me = mp.current_process()
        dd = me.stash["ddict"]

        alloc = System()
        num_tot_nodes = int(alloc.nnodes) 
        num_random_per_node = max(int(random_number/num_tot_nodes), 1)

        random_selection = []

        # Select num_random_per_node random keys
        current_host = host_id()
        manager_nodes = dd.manager_nodes
        key_list = []
        for i in range(len(manager_nodes)):
            if manager_nodes[i].h_uid == current_host:
                dm = dd.manager(i)
                key_list.extend(dm.keys())
                # Filter out keys containing model or iter info
                key_list = [key for key in key_list if "model" not in key and "iter" not in key]
       
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


def sort_dictionary_pg(dd: DDict, 
                       num_return_sorted: int, 
                       num_procs: int, 
                       nodelist, cdd: DDict, 
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
    
    logger.info(f"Direct sorting {direct_sort_num} keys per process")

    global_policy = Policy(distribution=Policy.Distribution.BLOCK)
    grp = ProcessGroup(policy=global_policy, pmi_enabled=True)

    logger.info(f"Launching sorting process group {nodelist}")
    for node in nodelist:
        node_name = Node(node).hostname
        local_policy = Policy(placement=Policy.Placement.HOST_NAME, 
                            host_name=node_name, 
                            cpu_affinity=list(range(0, 
                                                    max_num_procs_pn, 
                                                    max_num_procs_pn//num_procs_pn)))
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

