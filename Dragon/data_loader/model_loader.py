import os
from typing import Union
from dragon.data.ddict.ddict import DDict
from training.ST_funcs.smiles_regress_transformer_funcs import ModelArchitecture, ParamsJson

driver_path = os.getenv("DRIVER_PATH")

def save_model_weights(dd: Union[DDict, dict], model, model_iter: int, verbose = False):

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
            #dd[wkey] = weight
            if verbose:
                print(f"{wkey}: {weight.nbytes} bytes")
            tot_memory += weight.nbytes
    
    print(f"model weights: {num_layers=} {num_weights=} {tot_memory=}")

    # Future version will use broadcast put to send model to every manager
    dd.bput('model', weights_dict)
    dd.bput('model_iter', model_iter)

    #dd['model'] = weights_dict
    #dd['model_iter'] = model_iter

    print(f"Saved model {model_iter} to dictionary", flush=True)

def retrieve_model_from_dict(dd: Union[DDict, dict], fine_tune: bool = False):

    #weights_dict = dd["model"]
    #model_iter = dd["model_iter"]
    #hyper_params = dd["model_hyper_params"]

    weights_dict = dd.bget('model')
    model_iter = dd.bget("model_iter")
    hyper_params = dd.bget("model_hyper_params")


    try:
        model = ModelArchitecture(hyper_params, fine_tune=fine_tune).call()
    except Exception as e:
        print(f"Exception {e} raised in calling model")

    # Assign the weights back to the model
    for layer_idx, layer in enumerate(model.layers):
        weights = [weights_dict[f'model_layer_{layer_idx}_weight_{weight_idx}'] 
                    for weight_idx in range(len(layer.get_weights()))]
        layer.set_weights(weights)

    print(f"Finished loading model from dictionary\n")
    return model, model_iter, hyper_params

def load_pretrained_model(dd: Union[DDict, dict]):

    print("Loading pretrained model")
    # Read HyperParameters
    json_file = os.path.join(driver_path, "inference/config.json")
    hyper_params = ParamsJson(json_file)

    dd.bput('model_hyper_params', hyper_params)
    #dd['model_hyper_params'] = hyper_params

    print(f"Loaded hyper params: {hyper_params}", flush=True)
    # Load model and weights
    model = ModelArchitecture(hyper_params).call()
    model.load_weights(os.path.join(driver_path,"inference/smile_regress.autosave.model.h5"))
    print(f"Loaded pretrained model weights from disk", flush=True)
    save_model_weights(dd, model, model_iter=0, verbose=True)

    print(f"Loaded pretrained model into dictionary", flush=True)

if __name__ == "__main__":
    dd = {}
    load_pretrained_model(dd)
    model,model_iter,hyper_params = retrieve_model_from_dict(dd)
    print(f"{model=} {model_iter=} {hyper_params=}")
