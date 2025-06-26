import os

import dragon
import multiprocessing as mp
from dragon.native.process_group import ProcessGroup
from dragon.native.process import Process, ProcessTemplate, MSG_PIPE, MSG_DEVNULL
from dragon.infrastructure.connection import Connection
from dragon.data.ddict import DDict
from dragon.infrastructure.policy import Policy
from dragon.native.machine import Node

from .docking_openeye import run_docking


def launch_docking_sim(sim_dd, 
                        model_list_dd, 
                        num_procs, 
                        nodelist, 
                        continue_event=None):
    """Launch docking simulations

    :param cdd: Dragon distributed dictionary for top candidates
    :type dd: DDict
    :param num_procs: number of processes to use for docking
    :type num_procs: int
    """
    num_nodes = len(nodelist)
    num_procs_pn = num_procs//num_nodes
    run_dir = os.getcwd()

    skip_threads = os.getenv("SKIP_THREADS")
    if skip_threads:
        print(f"skipping threads {skip_threads}",flush=True)
        skip_threads = skip_threads.split(',')
        skip_threads = [int(t) for t in skip_threads]
    else:
        skip_threads = []


    cpu_affinity_string = os.getenv("CPU_AFFINITY")
    # cpu_affinity_string is of the form: "list:0-2,8-10:3-5,11-13"
    cpu_ranges = cpu_affinity_string.split(":")
    inf_cpu_bind = []
    barrier = None

    proc_count = 0
    for node_num in range(num_nodes):
        for proc in range(num_procs_pn):
            # Skip skip threads and threads bound to inference gpus
            if proc in skip_threads or proc in inf_cpu_bind:
                print(f"Skipping thread {proc} for docking",flush=True)
                continue
            else:
                proc_count += 1


    if continue_event is not None:
        for cr in cpu_ranges[1:]:
            bind_threads = []
            thread_ranges = cr.split(",")
            for tr in thread_ranges:
                t = tr.split("-")
                if len(t) == 1:
                    bind_threads.append(int(t[0]))
                elif len(t) == 2:
                    start_t = int(t[0])
                    end_t = int(t[1])
                    for st in range(start_t, end_t + 1):
                        bind_threads.append(st)
            inf_cpu_bind += bind_threads
        barrier = mp.Barrier(parties=proc_count,)
    
        
    print(f"Docking Sims using {proc_count} processes", flush=True)
    

    # Create the process group
    global_policy = Policy(distribution=Policy.Distribution.BLOCK)
    grp = ProcessGroup(policy=global_policy)
    for node_num in range(num_nodes):
        node_name = Node(nodelist[node_num]).hostname
        for proc in range(num_procs_pn):
            if proc in skip_threads or proc in inf_cpu_bind:
                continue
            proc_id = node_num*num_procs_pn+proc
            print(f"{proc_id} on {node_name} using proc {proc}", flush=True)
            local_policy = Policy(placement=Policy.Placement.HOST_NAME,
                                  host_name=node_name,
                                  cpu_affinity=[proc])
            grp.add_process(nproc=1,
                            template=ProcessTemplate(target=run_docking,
                                                        args=(sim_dd,
                                                            model_list_dd,
                                                            proc_id,
                                                            num_procs,
                                                            barrier,
                                                            continue_event,), 
                                                        cwd=run_dir,
                                                        policy=local_policy,
                                                        )
                            )

    # Launch the ProcessGroup
    grp.init()
    grp.start()
    print(f"Starting Process Group for Docking Sims on {num_procs} procs", flush=True)
    grp.join()
    print(f"Joined Process Group for Docking Sims",flush=True)
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
    if barrier is not None:
        model_list_dd.bput('simulated_compounds', list(sim_dd.keys()))
    

