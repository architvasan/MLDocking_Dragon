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
import logging
from .ST_funcs.smiles_regress_transformer_funcs import train_val_data, assemble_callbacks
from data_loader.model_loader import retrieve_model_from_dict, save_model_weights
import sys
import os
from time import perf_counter
import time

import dragon
from dragon.data.ddict.ddict import DDict
from logging_config import train_logger as logger


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

    fine_tune_log = f"training_{iter}.log"
    tic_start = perf_counter()

    prev_top_candidates = []
    training_iter = 0
    while continue_training(continue_event, training_iter):

        try:
            current_sort_list = model_list_dd.bget("current_sort_list")
            top_candidates = current_sort_list['smiles']
        except:
            top_candidates = []

        logger.info(f"current_sort_list has {len(top_candidates)} candidates")
        
        if top_candidates == prev_top_candidates:
            prev_top_candidates = top_candidates.copy()
            time.sleep(list_poll_interval_sec)
        else:
            ######## Build model #############
            prev_top_candidates = top_candidates.copy()

            if len(top_candidates) <= 10:
                logger.info("Too few candidates to train, skipping this iteration")
                logger.info(f"Current top candidates: {top_candidates}")
                training_iter += 1
                continue

            model, hyper_params = retrieve_model_from_dict(model_list_dd,)
            model_iter = model_list_dd.checkpoint_id

            #for layer in model.layers:
            #    if layer.name not in ['dropout_3', 'dense_3', 'dropout_4', 'dense_4', 'dropout_5', 'dense_5', 'dropout_6', 'dense_6']:
            #        layer.trainable = False

            logger.info(f"Create training data")
            ########Create training and validation data##### 
            x_train, y_train, x_val, y_val = train_val_data(sim_dd, method="stratified")
            logger.info(f"Finished creating training data")

            ######## Create callbacks #######
            callbacks = assemble_callbacks(hyper_params)
            
            # Only train if there is new data
            if len(x_train) > 0:
                logger.info(f"{BATCH=} {EPOCH=} {len(x_train)=}")
                
                with open(fine_tune_log, 'a') as sys.stdout:
                    history = model.fit(
                                x_train,
                                y_train,
                                batch_size=BATCH,
                                epochs=EPOCH,
                                verbose=2,
                                validation_data=(x_val,y_val),
                                callbacks=callbacks,
                            )
                    print("model fitting complete")
                sys.stdout = sys.__stdout__
                logger.info("model fitting complete")
                
                # Save to dictionary
                model_list_dd.checkpoint()
                save_model_weights(model_list_dd, model)
                model_iter = model_list_dd.checkpoint_id
                logger.info(f"Saved model weights to dictionary {model_iter=}")
                if save_model:
                    model_path = "current_model.keras"
                    model.save(model_path)
                    logger.info(f"{model_iter=} {model_path=}")
                logger.info("Saved fine tuned model to dictionary")
                if new_model_event is not None:
                    new_model_event.set()
                    logger.info("Setting new_model_event and waiting for barrier")
                    barrier.wait()
                    logger.info("Barrier passed, clearing new_model_event")
                    new_model_event.clear()
        training_iter += 1
            


