import os
from time import perf_counter

import dragon
import logging
import multiprocessing as mp
from dragon.native.process_group import ProcessGroup
from dragon.native.process import Process, ProcessTemplate, MSG_PIPE, MSG_DEVNULL
from dragon.infrastructure.connection import Connection
from dragon.data.ddict.ddict import DDict
from dragon.infrastructure.policy import Policy
from dragon.native.machine import Node
from logging_config import train_logger as logger
from driver_functions import get_gpu_affinity

from .smiles_regress_transformer_run import fine_tune


def launch_training(model_list_dd: DDict, 
                    node, 
                    sim_dd: DDict,
                    BATCH, 
                    EPOCH,
                    max_iter=None,
                    #continue_event=None,
                    stop_event=None,
                    new_model_event=None,
                    barrier=None,):
    """Launch the inference ruotine

    :param dd: Dragon distributed dictionary
    :type dd: DDict
    :param num_procs: number of processes to use for inference
    :type num_procs: int
    """

    sequential_workflow = stop_event is None
    logger.info(f"Current checkpoint is {model_list_dd.checkpoint_id}")
    run_dir = os.getcwd()

    tic = perf_counter()
    global_policy = Policy(distribution=Policy.Distribution.BLOCK)
    grp = ProcessGroup(policy=global_policy)
    node_name = Node(node).hostname

    inf_gpu_bind, inf_cpu_bind = get_gpu_affinity()
    num_procs_pn = len(inf_gpu_bind)  # number of procs per node is number of gpus

    # Bind the training to the first gpu on the assigned node
    logger.info(f"Training on node {node_name}")
    logger.info(f"Training bound to cpu cores {inf_cpu_bind[0]} and gpu {inf_gpu_bind[0]}")
    local_policy = Policy(placement=Policy.Placement.HOST_NAME, 
                          host_name=node_name, 
                          cpu_affinity=inf_cpu_bind[0], 
                          gpu_affinity=inf_gpu_bind[0],)
    grp.add_process(nproc=1, 
                    template=ProcessTemplate(target=fine_tune,
                                                args=(model_list_dd,
                                                    sim_dd, 
                                                    stop_event,
                                                    #continue_event,
                                                    new_model_event,
                                                    barrier, 
                                                    BATCH, EPOCH,
                                                    max_iter,
                                                    ), 
                                                cwd=run_dir,
                                                policy=local_policy, 
                                                ))
    
    # Launch the ProcessGroup 
    logger.info(f"Starting Process Group for training")
    grp.init()
    grp.start()
    logger.info(f"Training process group launched")
    if sequential_workflow:
        logger.info(f"Joining training process group")
        grp.join()
        grp.close()
    else:
        logger.info(f"Waiting for stop event to terminate training process group")
        stop_event.wait()
        grp.close()
    toc = perf_counter()
    logger.info(f"Training process group stopped in {toc-tic} seconds")


