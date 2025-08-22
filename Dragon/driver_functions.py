import logging
from dragon.data.ddict import DDict
from dragon.native.machine import System, Node
from logging_config import driver_logger as logger


def save_candidates(cdd: DDict, iter: int):

    candidate_list = cdd['current_sort_list']
    candidate_smiles = candidate_list["smiles"]
    candidate_pred = candidate_list["inf"]

    with open(f'top_candidates_{iter}.out','w') as f:
        f.write("# smiles  inf_score\n")
        for i in range(len(candidate_smiles)):
            smiles = candidate_smiles[i]
            pred = candidate_pred[i]
            #inference_scores = results['inf_scores']
            #print(inference_scores,flush=True)
            #docking_score = results['dock_score']
            line = f"{smiles}    {pred}"
            #for inf_result in inference_scores:
            #    sc = inf_result[0] # inference score
            #    mi = inf_result[1] # corresponding model iter
            #    line += f'{sc}    {mi}    '
            f.write(line+"\n")

def save_simulations(sdd: DDict, iter: int):

    simulated_smiles = sdd['simulated_compounds']

    with open(f'simulated_compounds_{iter}.out','w') as f:
        f.write("# smiles  dock_score\n")
        for i in range(len(simulated_smiles)):
            smiles = simulated_smiles[i]
            dock_score = sdd[smiles]["dock_score"]
            #pred = candidate_pred[i]
            line = f"{smiles}    {dock_score}"
            f.write(line+"\n")

def max_data_dict_size(num_keys: int, 
                       node_counts: dict,
                       model_size=33, 
                       smiles_key_val_size=14.6, 
                       canidate_sim_size_per_iter=1.5, 
                       max_pool_frac=0.8):

    logger.info(f"Estimating dictionary sizes with a maximum data pool utiliztion of {max_pool_frac*100} per cent")

    # Get information about the allocation
    alloc = System()
    num_tot_nodes = int(alloc.nnodes)
    
    # Two sources of data in data dictionary
    # Smiles data: approx 14.6 MB per file
    # Trained model: approximately 33 MB, broadcast one copy to each node
    # Data needed is 33 MB*num_nodes + num_keys*14.6
    min_data_req = smiles_key_val_size*num_keys

    # Assume we want to store 10 top cand lists and associated simulation results
    min_sim_dict_size = 10.*canidate_sim_size_per_iter

    min_model_dict_size = model_size*num_tot_nodes

    # Assume you need 1-max_pool_frac per cent overhead in data dictionary
    data_dict_size = min_data_req/(max_pool_frac)
    sim_dict_size = min_sim_dict_size/(max_pool_frac)
    model_dict_size = min_model_dict_size/(max_pool_frac)

    # Convert from MB to GB
    sim_dict_size /= 1024
    data_dict_size /= 1024
    model_dict_size /= 1024

    # Ensure there is a minimum of 1 GB per node
    sim_dict_size = max(sim_dict_size, node_counts["simulation"])
    data_dict_size = max(data_dict_size, node_counts["inference"])
    model_dict_size = max(model_dict_size, num_tot_nodes)

    max_mem = ddict_mem_check(mem_fraction=max_pool_frac)

    logger.info(f"Memory available for ddicts: {max_mem} GB")

    if sim_dict_size + data_dict_size + model_dict_size > max_mem:
        raise Exception(f"Not enough mem for dictionaries: {max_mem=} {max_pool_frac=} {data_dict_size=} {model_dict_size=} {sim_dict_size=}")

    return int(data_dict_size), int(sim_dict_size), int(model_dict_size)


def ddict_mem_check(mem_fraction=0.5):

    # let's place the DDict across all nodes Dragon is running on
    my_system = System()

    total_mem = 0
    for huid in my_system.nodes:
        anode = Node(huid)
        total_mem += anode.physical_mem
    dict_mem = mem_fraction * total_mem  # use fraction of the mem

    # Convert dict_mem to GB
    dict_mem /= 1024*1024*1024

    return int(dict_mem)

def get_prime_number(n: int):
    # Use 1, 2 or 3 if that is n
    if n <= 3:
        return n
    
    # All prime numbers are odd except two
    if not (n & 1):
        n -= 1
    
    for i in range(n, 3, -2):
        isprime = True
        for j in range(3,n):
            if i % j == 0:
                isprime = False
                break
        if isprime:
            return i
    return 3

