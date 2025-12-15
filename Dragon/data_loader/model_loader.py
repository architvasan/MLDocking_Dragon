import logging
import os
from typing import Union
from dragon.data.ddict.ddict import DDict
from training.ST_funcs.smiles_regress_transformer_funcs import ModelArchitecture, ParamsJson
from time import perf_counter

driver_path = os.getenv("DRIVER_PATH")

from logging_config import load_logger as logger

def save_model_weights(dd: Union[DDict, dict], model, checkpoint=False):

    # Create dictionary of model weights
    weights_dict = {}
    num_layers = 0
    num_weights = 0
    tot_memory = 0
    for layer_idx, layer in enumerate(model.layers):
        num_layers += 1
        for weight_idx, weight in enumerate(layer.get_weights()):
            num_weights += 1
            # Create a key for each weight
            wkey = f'model_layer_{layer_idx}_weight_{weight_idx}'
            # Save the weight in the dictionary
            weights_dict[wkey] = weight
            logger.debug(f"{wkey}: {weight.nbytes} bytes")
            tot_memory += weight.nbytes
    
    logger.debug(f"model weights: {num_layers=} {num_weights=} {tot_memory=}")

    # Checkpoint if requested
    if checkpoint:
        dd.checkpoint()

    # Use broadcast put to save model weights on every manager
    dd.bput(f'model', weights_dict)

    logger.info(f"Saved model to dictionary on checkpoint {dd.checkpoint_id}")

def retrieve_model_from_dict(dd: Union[DDict, dict], checkpoint=False, hyper_params=None):
 
    logger.debug("Retrieving model from dictionary")
    if checkpoint:
        logger.debug("Checkpointing model dictionary")
        dd.checkpoint()

    model_iter = dd.checkpoint_id
    logger.debug(f"Loading model weights from dictionary with checkpoint_id: {model_iter}")
    logger.info(f"{list(dd.keys())=}")

    tic = perf_counter()
    # This will block until model weight are available
    weights_dict = dd.bget('model')
    logger.debug(f"bget time: {perf_counter()-tic:.4f} seconds")
    if hyper_params is None:
        hyper_params = dd["model_hyper_params"]

    logger.debug(f"Loaded {len(weights_dict)} weights from dictionary")

    model = ModelArchitecture(hyper_params).call()

    logger.debug(f"Called model")
    
    # Assign the weights back to the model
    for layer_idx, layer in enumerate(model.layers):
        weights = [weights_dict[f'model_layer_{layer_idx}_weight_{weight_idx}'] 
                    for weight_idx in range(len(layer.get_weights()))]
        layer.set_weights(weights)

    logger.info(f"Finished loading model from dictionary")
    return model, hyper_params

def load_pretrained_model(dd: Union[DDict, dict]):

    logger.info("Loading pretrained model")
    
    # Read HyperParameters
    json_file = os.path.join(driver_path, "inference/config.json")
    hyper_params = ParamsJson(json_file)

    # Use persistent put to save hyper parameters (they do not change)
    dd.pput('model_hyper_params', hyper_params)
    logger.info(f"Loaded hyper params: {hyper_params}")

    # Load model and weights
    model = ModelArchitecture(hyper_params).call()
    model.load_weights(os.path.join(driver_path,"inference/smile_regress.autosave.model.h5"))
    logger.info(f"Loaded pretrained model weights from disk")
    save_model_weights(dd, model)

    logger.info(f"Loaded pretrained model into dictionary")

if __name__ == "__main__":
    dd = {}
    load_pretrained_model(dd)
    model,hyper_params = retrieve_model_from_dict(dd)
    print(f"{model=} {hyper_params=}")
