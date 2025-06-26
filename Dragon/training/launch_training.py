import os

import dragon
import multiprocessing as mp
from dragon.native.process_group import ProcessGroup
from dragon.native.process import Process, ProcessTemplate, MSG_PIPE, MSG_DEVNULL
from dragon.infrastructure.connection import Connection
from dragon.data.ddict.ddict import DDict
from dragon.infrastructure.policy import Policy
from dragon.native.machine import Node

from .smiles_regress_transformer_run import fine_tune


def launch_training(model_list_dd: DDict, 
                    node, 
                    sim_dd: DDict,
                    BATCH, 
                    EPOCH,
                    continue_event=None,
                    new_model_event=None,
                    barrier=None,):
    """Launch the inference ruotine

    :param dd: Dragon distributed dictionary
    :type dd: DDict
    :param num_procs: number of processes to use for inference
    :type num_procs: int
    """
    run_dir = os.getcwd()

    # Create the process group
    global_policy = Policy(distribution=Policy.Distribution.BLOCK)
    grp = ProcessGroup(policy=global_policy)

    node_name = Node(node).hostname

    gpu_devices_string = os.getenv("GPU_DEVICES")

    inf_gpu_bind = []
    for g in gpu_devices_string.split(","):
        if "." in g:
            inf_gpu_bind.append([float(g)])
        else:
            inf_gpu_bind.append([int(g)])
    num_procs_pn = len(inf_gpu_bind)  # number of procs per node is number of gpus

    cpu_affinity_string = os.getenv("CPU_AFFINITY")
    cpu_ranges = cpu_affinity_string.split(":")
    inf_cpu_bind = []
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
        inf_cpu_bind.append(bind_threads)

    # Bind the training to the first gpu on the assigned node

    local_policy = Policy(placement=Policy.Placement.HOST_NAME, 
                          host_name=node_name, 
                          cpu_affinity=inf_cpu_bind[0], 
                          gpu_affinity=inf_gpu_bind[0],)
    grp.add_process(nproc=1, 
                    template=ProcessTemplate(target=fine_tune,
                                                args=(model_list_dd,
                                                    sim_dd, 
                                                    continue_event,
                                                    new_model_event,
                                                    barrier, 
                                                    BATCH, EPOCH, 
                                                    ), 
                                                cwd=run_dir,
                                                policy=local_policy, 
                                                ))
    
    # Launch the ProcessGroup 
    grp.init()
    grp.start()
    print(f"Starting Process Group for Training")
    
    grp.join()
    grp.close()
    print(f"Training process group stopped",flush=True)
    #print(dd["model_iter"])
    #print(dd["model"])

