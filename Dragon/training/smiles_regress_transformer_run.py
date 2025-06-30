# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import backend as K
# from tensorflow.keras import layers
# from tensorflow.keras.callbacks import (
#     CSVLogger,
#     EarlyStopping,
#     ModelCheckpoint,
#     ReduceLROnPlateau,
# )

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence, text

#from .ST_funcs.clr_callback import *
from .ST_funcs.smiles_regress_transformer_funcs import train_val_data
from data_loader.model_loader import retrieve_model_from_dict, save_model_weights
import sys
import os
import time

import dragon
from dragon.data.ddict.ddict import DDict


def continue_training(continue_event, training_iter):
    if continue_event is None:
        if training_iter == 0:
            return True
        else:
            return False
    else:
        return continue_event.is_set()

def fine_tune(model_list_dd: DDict, 
                sim_dd: DDict,
                continue_event,
                new_model_event,
                barrier, 
                BATCH: int, 
                EPOCH: int, 
                save_model=True,
                list_poll_interval_sec=60):

    fine_tune_log = "training.log"

    prev_top_candidates = []
    training_iter = 0
    while continue_training(continue_event, training_iter):

        if "current_sort_list" in model_list_dd.keys():
            top_candidates = model_list_dd.bget("current_sort_list")
        else:
            top_candidates = []

        if top_candidates == prev_top_candidates:
            time.sleep(list_poll_interval_sec)
        else:
            ######## Build model #############
                
            model, hyper_params = retrieve_model_from_dict(model_list_dd,)
            model_iter = model_list_dd.current_checkpoint_id

            for layer in model.layers:
                if layer.name not in ['dropout_3', 'dense_3', 'dropout_4', 'dense_4', 'dropout_5', 'dense_5', 'dropout_6', 'dense_6']:
                    layer.trainable = False

            with open(fine_tune_log, 'a') as f:
                f.write(f"Create training data\n")
            ########Create training and validation data##### 
            x_train, y_train, x_val, y_val = train_val_data(sim_dd)
            with open(fine_tune_log, 'a') as f:
                f.write(f"Finished creating training data\n")
            
            # Only train if there is new data
            if len(x_train) > 0:
                with open(fine_tune_log, 'a') as f:
                    f.write(f"{BATCH=} {EPOCH=} {len(x_train)=}\n")
                
                with open(fine_tune_log, 'a') as sys.stdout:
                    history = model.fit(
                                x_train,
                                y_train,
                                batch_size=BATCH,
                                epochs=EPOCH,
                                verbose=2,
                                validation_data=(x_val,y_val),
                                #callbacks=callbacks,
                            )
                    print("model fitting complete",flush=True)
                sys.stdout = sys.__stdout__
                print("model fitting complete",flush=True)
                
                
                # Save to dictionary
                model_list_dd.checkpoint()
                save_model_weights(model_list_dd, model)
                model_iter = model_list_dd.current_checkpoint_id
                print(f"Saved model weights to dictionary {model_iter=}", flush=True)
                if save_model:
                    model_path = "current_model.keras"
                    model.save(model_path)
                    with open("model_iter",'w') as f:
                        f.write(f"{model_iter=} {model_path=}")
                print("Saved fine tuned model to dictionary",flush=True)
                if new_model_event is not None:
                    new_model_event.set()
                    barrier.wait()
                    new_model_event.clear()
        training_iter += 1
            


