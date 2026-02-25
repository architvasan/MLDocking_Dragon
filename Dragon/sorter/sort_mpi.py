import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI
from time import perf_counter
from dragon.utils import host_id
from dragon.data.ddict import DDict
from logging_config import sort_logger as logger
from dragon.globalservices.api_setup import connect_to_infrastructure
connect_to_infrastructure()


def mpi_sort(_dict: DDict, num_keys: int, num_return_sorted: int, candidate_dict: DDict):
    MPI.Init()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
        
    checkpoint_id = candidate_dict.checkpoint_id
    #print(f"Sort rank {rank} has started",flush=True)
    if rank == 0: logger.info(f"MPI Sorting starting on {size} ranks")
    
    tic = perf_counter()
    
    current_host = host_id()
    manager_nodes = _dict.manager_nodes
    #stats = _dict.stats
    #cpu_num = psutil.Process().cpu_num()
    #print(f"Sort rank {rank} on host {current_host} {cpu_num=}",flush=True)

    # Get local keys
   
    stic = perf_counter()
    # Get the host of every rank; assume all hosts are manager nodes
    all_hosts = []
    for h in manager_nodes:
        if h.h_uid not in all_hosts:
            all_hosts.append(h.h_uid)
    
    if rank == 0:
        logger.info(f"Sorting on {len(all_hosts)} nodes")

    # Create communicator for local ranks only
    ctic = perf_counter()
    for h in all_hosts:
        color = True if current_host == h else MPI.UNDEFINED
        new_comm = comm.Split(color=color, key=rank)
        if current_host == h:
            my_host_comm = new_comm
    local_rank = my_host_comm.Get_rank()
    local_size = my_host_comm.Get_size()
    
    stoc = perf_counter()
    local_comm_time = stoc - stic

    #print(f"Sort rank {rank} has local rank {local_rank} on host {current_host}; setup time is {stoc-stic}; comm setup time is {stoc-ctic}")
    
    # Get keys on local rank and bcast them to other local ranks
    key_list = []
    for i in range(len(manager_nodes)):
        # local rank 0 gets keys from local managers
        if local_rank == 0:
            if manager_nodes[i].h_uid == current_host:
                ktic = perf_counter()
                dm = _dict.manager(i)
                key_list.extend(dm.keys())
                # Filter out keys containing model or iter info
                key_list = [key for key in key_list if "model" not in key and "iter" not in key]
                ktoc = perf_counter()
                #print(f"Sort rank {rank} retrieved local keys in {ktoc - ktic} seconds",flush=True)
    
    if local_rank == 0:
        my_key_list = []
        direct_sort_num = len(key_list)//local_size
        logger.info(f"Sort rank {rank} retrieved {len(key_list)} local keys")
        for i in range(local_size):
            min_index = i*direct_sort_num
            max_index = min(min_index + direct_sort_num, num_keys)
            my_key_list.append(key_list[min_index:max_index])
    else:
        my_key_list = None
    # Communicate key_list to local ranks
    ctic = perf_counter()
    my_key_list = my_host_comm.scatter(my_key_list,root=0)
    ctoc = perf_counter()
    
    toc = perf_counter()
    setup_time = toc - tic
    all_setup_times = comm.gather(setup_time, root=0)
    if rank == 0:
        ave_setup_time = sum(all_setup_times) / len(all_setup_times)
        logger.info(f"Average key and comm setup time is {ave_setup_time:.3f} seconds")

    #print(f"Sort rank {rank} got keys in {toc-tic} seconds, sorting {len(my_key_list)} local keys",flush=True)

    # Direct sort keys assigned to this rank
    tic = perf_counter()
    my_results = []
    
    for i,key in enumerate(my_key_list):
        #print(f"Sort rank {rank} getting key {key} on iter {i}",flush=True)
        val = _dict[key]
        # Skip values from models before checkpoint_id
        if val['model_iter'] < checkpoint_id:
            continue
        #print(f"Sort rank {rank} finished getting key {key} on iter {i}",flush=True)

        # Only include this key if it has non-zero inf values
        if any(val["inf"]):
            num_smiles = len(val['inf'])
            this_value = list(zip(val["inf"],
                                  val["smiles"],
                                  [val["model_iter"] for _ in range(num_smiles)]))
            my_results.extend(this_value)
            my_results.sort(key=lambda tup: tup[0])
            my_results = my_results[-num_return_sorted:]
     
    my_results.sort(key=lambda tup: tup[0])
    my_results = my_results[-num_return_sorted:]

    toc = perf_counter()

    direct_sort_time = toc - tic
    all_direct_sort_times = comm.gather(direct_sort_time, root=0)
    if rank == 0:
        ave_direct_sort_time = sum(all_direct_sort_times) / len(all_direct_sort_times)
        logger.info(f"Average direct sort time is {ave_direct_sort_time:.3f} seconds")
    #print(f"Rank {rank} finished direct sort in {toc-tic} seconds; found {len(my_results)} results; starting local merge",flush=True)
            
    # Combine local results
    com_tic = perf_counter()
    local_results = []
    num_my_results = len(my_results)
    for i in range(num_return_sorted):
    #for i,res in enumerate(my_results[::-1]):
        tic = perf_counter()
        if i < num_my_results:
            res = my_results[::-1][i]
        else:
            res = (0.0,'dummy',-1)
        gathered_local_results = my_host_comm.gather(res, root=0)
        toc = perf_counter()
        if local_rank == 0:
            continue_loop = True
            #print(f"Sort rank {rank} gathered in {toc-tic} seconds on iter {i}",flush=True)
            last_results = local_results.copy()
            local_results.extend(gathered_local_results)
            local_results.sort(key=lambda tup: tup[0])
            local_results = local_results[-num_return_sorted:]
            # Since we are pulling results in reverse order from all lists,
            # once adding new entries does not change the trucated list
            # there are no more entries that will make it in the top candidates.
            # The local_rank can stop gathering results.
            if last_results == local_results:
                #print(f"Rank {rank} exiting local merge",flush=True)
                continue_loop = False
        else:
            assert gathered_local_results is None
            local_results = None
            continue_loop = True
        continue_loop = my_host_comm.bcast(continue_loop, root=0)
        if not continue_loop:
            break
    #print(f"Rank {rank} finished node local merge",flush=True)

    # Merge results between nodes
    # Make a communicator that includes all ranks with local_rank=0    
    
    color = True if local_rank == 0 else MPI.UNDEFINED
    root_comm = comm.Split(color=color, key=rank)
    root_rank = None
    #print(f"Rank {rank} starting node global merge",flush=True)
    if local_rank == 0:
        root_size = root_comm.Get_size()
        root_rank = root_comm.Get_rank()
        #print(f"Rank {rank} has {local_rank=} and {root_rank=}",flush=True)
        all_results = []
        num_local_results = len(local_results)
        for i in range(num_return_sorted):
            if i < num_local_results:
                res = local_results[::-1][i]
            else:
                res = (0.0,'dummy',-1)
            #print(f"Rank {rank} gathering results for iter {i}",flush=True)
            gathered_node_results = root_comm.gather(res, root=0)
            #print(f"Rank {rank} finished gather for iter {i}; is None? {gathered_node_results==None}",flush=True)
            if root_rank == 0:
                continue_loop = True
                last_results = all_results.copy()
                all_results.extend(gathered_node_results)
                all_results.sort(key=lambda tup: tup[0])
                all_results = all_results[-num_return_sorted:]
                if last_results == all_results:
                    #print(f"Rank {rank} exiting node merge after iter {i}",flush=True)
                    continue_loop = False
            else:
                assert gathered_node_results is None
                all_results = None
                continue_loop = True
            continue_loop = root_comm.bcast(continue_loop, root=0)
            if not continue_loop:
                break
    
    if root_rank == 0: logger.info(f"Rank {rank} moving onto saving")
    com_toc = perf_counter()
    if root_rank == 0:
        com_time = com_toc - com_tic
        logger.info(f"Collected sorted results on rank 0 in {com_time:.3f} seconds")
        #print(f"{my_results=}")
        # put data in candidate_dict
        
        #top_candidates = all_results
        # filter out any 0 values or dummy values
        logger.info(f"Number of results {len(all_results)=}")
        #print(all_results,flush=True)
        top_candidates = [c for c in all_results if c[0] > 0 and c[1] != 'dummy']
        num_top_candidates = len(top_candidates)
        with open("sort_controller.log", "a") as f:
            f.write(f"Collected {num_top_candidates=}\n")
        logger.info(f"Collected {num_top_candidates=}")
        if num_top_candidates > 0:
            
            # current_sort_iter = candidate_dict.bget("current_sort_iter")
            # if current_sort_iter > -1:
            #     current_sort_list = candidate_dict.bget("current_sort_list")
            #     candidate_dict[str(current_sort_iter)] = current_sort_list
            
            candidate_inf,candidate_smiles,candidate_model_iter = zip(*top_candidates)
            non_zero_infs = len([cinf for cinf in candidate_inf if cinf != 0])
            
            logger.info(f"Sorted list contains {non_zero_infs} non-zero inference results out of {len(candidate_inf)}")
            sort_val = {"inf": list(candidate_inf), "smiles": list(candidate_smiles), "model_iter": list(candidate_model_iter)}
        
            save_list(candidate_dict, sort_val, logger)    
    #print(f"Rank {rank} done",flush=True)
    MPI.Finalize()
    
    return

def save_list(candidate_dict, sort_val, logger):

    logger.info("HERE IS THE CANDIDATE LIST (first 10 only)")
    logger.info("******************************************")
    sliced = sort_val["smiles"][:10]
    logger.info(f"{sliced}")
    candidate_dict.bput("current_sort_list", sort_val)
    #candidate_dict.bput("current_sort_iter", ckey)
    logger.info(f"candidate dictionary saved")

