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
                        #continue_event=None,
                        stop_event=None,
                        new_model_event=None,
                        checkpoint_barrier=None):
    """Launch docking simulations
    :param sim_dd: Dragon distributed dictionary for simulation results
    :type dd: DDict
    :param model_list_dd: Dragon distributed dictionary for model list and checkpoints
    :type model_list_dd: DDict
    :param num_procs: number of processes to use for docking
    :type num_procs: int
    :param nodelist: list of node hostnames to use
    :type nodelist: list of str
    :param thread_list: list of CPU threads to bind processes to
    :type thread_list: list of int
    :param continue_event: multiprocessing event to signal whether to continue simulations
    :type continue_event: mp.Event
    :param new_model_event: multiprocessing event to signal new model is available
    :type new_model_event: mp.Event
    :param checkpoint_barrier: multiprocessing barrier to synchronize at checkpoints
    :type checkpoint_barrier: mp.Barrier
    """

    sequential_workflow = stop_event is None

    run_dir = os.getcwd()
    num_nodes = len(nodelist)
    num_procs_pn = len(thread_list)
    proc_count = num_nodes * num_procs_pn

    if sequential_workflow:
        update_barrier = None
    else:
        update_barrier = mp.Barrier(parties=proc_count,)

    logger.info(f"Current checkpoint is {model_list_dd.checkpoint_id}")
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
                                                            #continue_event,
                                                            stop_event,
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
    if sequential_workflow:
        grp.join()
        logger.info(f"Joined Process Group for Docking Sims")
        grp.close()
    else:
        stop_event.wait()
        grp.terminate()
        logger.info(f"Terminated Process Group for Docking Sims")

    # Update simulated compounds list for sequential workflow
    if sequential_workflow:
        simulated_compounds = list(sim_dd.keys())
        logger.info(f"Retrieved {len(simulated_compounds)} simulated compounds from sim_dd")
        model_list_dd.bput("simulated_compounds", list(sim_dd.keys()))
        logger.info(f"Returned simulated_compounds list to model_list_dd")

    #toc_write = perf_counter()
    toc = perf_counter()
    
    #run_time = max(run_times)
    #avg_io_time = (toc_write-tic_write) + sum(ddict_times)/len(ddict_times)
    #max_io_time = (toc_write-tic_write) + max(ddict_times)
    #print(f'Performed docking simulation: total={run_time}, IO_avg={avg_io_time}, IO_max={max_io_time}',flush=True)
    logger.info(f"Performed docking simulations in {toc-tic} seconds")    

