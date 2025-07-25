import os
import logging

import dragon
import multiprocessing as mp
from dragon.native.process_group import ProcessGroup
from dragon.native.process import Process, ProcessTemplate, MSG_PIPE, MSG_DEVNULL
from dragon.infrastructure.connection import Connection
from dragon.data.ddict import DDict
from dragon.infrastructure.policy import Policy
from dragon.native.machine import Node
import os
import sys

from inference.utils_transformer import ParamsJson, ModelArchitecture, pad
from logging_config import inf_logger as logger
from .run_inference import infer

driver_path = os.getenv("DRIVER_PATH")

#logger = setup_logger('inf', "inference.log", level=logging.INFO)
#logger = logging.getLogger('inference')
#logger.propagate = False  # Prevent logging from propagating to the root logger

#logger = logging.getLogger(__name__)

# logger = logging.getLogger('inference')
# handler = logging.StreamHandler()
# handler.setFormatter(logging.Formatter('INFERENCE %(levelname)s: %(asctime)s: %(message)s', 
#                                         datefmt='%m-%d-%Y %I:%M:%S %p'))
# logger.addHandler(handler)

def launch_inference(data_dd: DDict, 
                    model_list_dd: DDict, 
                    nodelist, 
                    num_procs: int, 
                    inf_num_limit: int,
                    continue_event = None,
                    new_model_event = None,
                    barrier=None,
                    debug=True):
    """Launch the inference routine

    :param dd: Dragon distributed dictionary
    :type dd: DDict
    :param num_procs: number of processes to use for inference
    :type num_procs: int
    """

    #logger = setup_logger('inf', "inference.log", level=logging.INFO)
    
    num_inf_nodes = len(nodelist)

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

    run_dir = os.getcwd()
    logger.info(f"{inf_cpu_bind=}")
    logger.info(f"{inf_gpu_bind=}")
    if len(inf_cpu_bind) != len(inf_gpu_bind):
        raise (Exception("Number of cpu bindings does not match the number of gpus"))

    # Get checkpoint id
    checkpoint_id = model_list_dd.checkpoint_id

    #bar = mp.Barrier(parties=num_inf_nodes * num_procs_pn)

    # Create the process group
    global_policy = Policy(distribution=Policy.Distribution.BLOCK)
    grp = ProcessGroup(policy=global_policy)
    for node_num in range(num_inf_nodes):
        node_name = Node(nodelist[node_num]).hostname
        for proc in range(num_procs_pn):
            proc_id = node_num * num_procs_pn + proc

            if continue_event is not None and proc_id == 0:
                # In the asynchronous workflow, the first gpu is reserved for the fine-tuning process
                continue

            local_policy = Policy(placement=Policy.Placement.HOST_NAME,
                                  host_name=node_name, 
                                  cpu_affinity=inf_cpu_bind[proc],
                                  gpu_affinity=inf_gpu_bind[proc])
            grp.add_process(nproc=1, 
                            template=ProcessTemplate(target=infer, 
                                                     args=(data_dd,
                                                        model_list_dd,
                                                        num_procs_pn,
                                                        proc_id, 
                                                        continue_event, # Continue event not used in sequential wf
                                                        checkpoint_id,
                                                        inf_num_limit,
                                                        debug,
                                                        new_model_event,
                                                        barrier,
                                                        ), 
                                                     cwd=run_dir,
                                                     policy=local_policy,))
    
    # Launch the ProcessGroup 

    grp.init()
    logger.info(f"Starting Process Group for Inference")
    grp.start()
    grp.join()
    logger.info(f"Joined Process Group for Inference")
    grp.close()
