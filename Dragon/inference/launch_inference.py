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

from inference.utils_transformer import ParamsJson, ModelArchitecture, pad
from logging_config import inf_logger as logger
from driver_functions import get_gpu_affinity
from .run_inference import run_inference_loop

driver_path = os.getenv("DRIVER_PATH")


def launch_inference(data_dd: DDict, 
                    model_list_dd: DDict, 
                    nodelist, 
                    num_procs: int, 
                    inf_num_limit: int,
                    stop_event=None,
                    #continue_event = None,
                    new_model_event = None,
                    barrier=None,):
    """Launch the inference routine
    :param data_dd: DDict containing the data for inference
    :type data_dd: DDict
    :param model_list_dd: DDict containing the model list for inference
    :type model_list_dd: DDict
    :param nodelist: List of nodes to use for inference
    :type nodelist: list
    :param num_procs: Number of processes to launch
    :type num_procs: int
    :param inf_num_limit: Limit on number of inference samples per worker
    :type inf_num_limit: int
    :param continue_event: Event to signal continuation of inference
    :type continue_event: mp.Event
    :param new_model_event: Event to signal new model availability
    :type new_model_event: mp.Event
    :param barrier: Barrier for synchronization
    :type barrier: mp.Barrier
    """

    sequential_workflow = stop_event is None
    
    logger.info(f"Current checkpoint is {model_list_dd.checkpoint_id}")

    num_inf_nodes = len(nodelist)
    
    inf_gpu_bind, inf_cpu_bind = get_gpu_affinity()
    num_procs_pn = len(inf_gpu_bind)  # number of procs per node is number of gpus
    logger.info(f"Inference running on {num_inf_nodes} nodes and {num_procs_pn} processes per node")
    if sequential_workflow:
        logger.info(f"Inference running in sequential workflow mode with {num_inf_nodes*num_procs_pn} total workers")
    else:
        logger.info(f"Inference running in asynchronous workflow mode with {num_inf_nodes*num_procs_pn - 1} total workers (reserving 1 GPU per node for training)")

    run_dir = os.getcwd()
    logger.info(f"{inf_cpu_bind=}")
    logger.info(f"{inf_gpu_bind=}")
    if len(inf_cpu_bind) != len(inf_gpu_bind):
        raise (Exception("Number of cpu bindings does not match the number of gpus"))

    # Create the process group
    tic = perf_counter()
    global_policy = Policy(distribution=Policy.Distribution.BLOCK)
    grp = ProcessGroup(policy=global_policy)
    for node_num in range(num_inf_nodes):
        node_name = Node(nodelist[node_num]).hostname
        for proc in range(num_procs_pn):
            proc_id = node_num * num_procs_pn + proc

            if not sequential_workflow and proc_id == 0:
                logger.info(f"Skipping inference launch on node {node_name} proc {inf_gpu_bind[proc]} to reserve GPU for training")    
                continue

            local_policy = Policy(placement=Policy.Placement.HOST_NAME,
                                  host_name=node_name, 
                                  cpu_affinity=inf_cpu_bind[proc],
                                  gpu_affinity=inf_gpu_bind[proc])
            grp.add_process(nproc=1, 
                            template=ProcessTemplate(target=run_inference_loop,
                                                    args=(model_list_dd,
                                                            data_dd,
                                                            proc_id,
                                                            num_procs_pn*num_inf_nodes,
                                                            #continue_event,
                                                            stop_event,
                                                            new_model_event,
                                                            barrier,), 
                                                     cwd=run_dir,
                                                     policy=local_policy,))
    
    # Launch the ProcessGroup 
    grp.init()
    logger.info(f"Starting Process Group for Inference")
    grp.start()
    if sequential_workflow:
        grp.join()
        logger.info(f"Joined Process Group for Inference")
        grp.close()
    else:
        stop_event.wait()
        try:
            grp.close()
        except Exception as e:
            logger.info(f"Inf process group closed with exception {e}")
        logger.info(f"Stop event has been set, closing inference process group")
        
    toc = perf_counter()
    logger.info(f"Performed inference in {toc-tic} seconds")
