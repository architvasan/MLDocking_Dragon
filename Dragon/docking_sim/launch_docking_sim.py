import os
import logging
from time import perf_counter

import dragon
import multiprocessing as mp
from dragon.native.process_group import ProcessGroup
from dragon.native.process import Process, ProcessTemplate, MSG_PIPE, MSG_DEVNULL
from dragon.infrastructure.connection import Connection
from dragon.data.ddict import DDict
from dragon.infrastructure.policy import Policy
from dragon.native.machine import Node

from .docking_openeye import run_docking

from logging_config import sim_logger as logger


def launch_docking_sim(sim_dd, 
                        model_list_dd, 
                        num_procs, 
                        nodelist,
                        thread_list,
                        continue_event=None,
                        new_model_event=None,
                        checkpoint_barrier=None):
    """Launch docking simulations
    :param sim_dd: Dragon distributed dictionary for simulation results
    :type dd: DDict
    :param num_procs: number of processes to use for docking
    :type num_procs: int
    """
    run_dir = os.getcwd()
    num_nodes = len(nodelist)
    num_procs_pn = len(thread_list)
    proc_count = num_nodes * num_procs_pn

    
    update_barrier = None
    if continue_event is not None:
        update_barrier = mp.Barrier(parties=proc_count,)
    
        
    logger.info(f"Docking Sims using {proc_count} processes")
    logger.info(f"Docking Sims using {num_procs_pn} processes per node on {num_nodes} nodes")
    logger.info(f"Docking Sims using threads {thread_list}")
    

    # Create the process group
    tic = perf_counter()
    global_policy = Policy(distribution=Policy.Distribution.BLOCK)
    grp = ProcessGroup(policy=global_policy)
    count_threads = 0
    for node_num in range(num_nodes):
        node_name = Node(nodelist[node_num]).hostname
        for proc in range(num_procs_pn):
            # if proc in skip_threads or proc in inf_cpu_bind:
            #     continue
            proc_id = node_num*num_procs_pn+proc
            count_threads += 1
            logger.debug(f"{proc_id} on {node_name} using proc {proc}")
            local_policy = Policy(placement=Policy.Placement.HOST_NAME,
                                  host_name=node_name,
                                  cpu_affinity=[thread_list[proc]],)
            grp.add_process(nproc=1,
                            template=ProcessTemplate(target=run_docking,
                                                        args=(sim_dd,
                                                            model_list_dd,
                                                            proc_id,
                                                            num_procs,
                                                            update_barrier,
                                                            continue_event,
                                                            new_model_event,
                                                            checkpoint_barrier,), 
                                                        cwd=run_dir,
                                                        policy=local_policy,
                                                        stdout=MSG_PIPE
                                                        )
                            )

    # Launch the ProcessGroup
    logger.info(f"Starting Process Group for Docking Sims on {num_procs} procs")
    logger.debug(f"Counted {count_threads} processes in process group")
    grp.init()
    grp.start()
    
    grp.join()
    logger.info(f"Joined Process Group for Docking Sims")
    grp.close()

    # Collect candidate keys and save them to simulated keys
    # Lists will have a key that is a digit
    # Non-smiles keys that are not digits are -1, max_sort_iter and simulated_compounds
    # simulated_compounds = [k for k in cdd.keys() if not k.isdigit() and 
    #                                                 k != '-1' and 
    #                                                 "iter" not in k and
    #                                                 "current" not in k and
    #                                                 k != "simulated_compounds" and 
    #                                                 k != "random_compound_sample"]
    if update_barrier is not None:
        model_list_dd.bput('simulated_compounds', list(sim_dd.keys()))
    tic_write = perf_counter()
    simulated_compounds = [k for k in sdd.keys() if not k.isdigit() and 
                                                    k != '-1' and 
                                                    "iter" not in k and
                                                    "current" not in k and
                                                    k != "simulated_compounds" and 
                                                    k != "random_compound_sample"]
    sdd.bput('simulated_compounds', simulated_compounds)
    toc_write = perf_counter()
    toc = perf_counter()
    
    #run_time = max(run_times)
    #avg_io_time = (toc_write-tic_write) + sum(ddict_times)/len(ddict_times)
    #max_io_time = (toc_write-tic_write) + max(ddict_times)
    #print(f'Performed docking simulation: total={run_time}, IO_avg={avg_io_time}, IO_max={max_io_time}',flush=True)
    logger.info(f"Performed docking simulations in {toc-tic} seconds")    

