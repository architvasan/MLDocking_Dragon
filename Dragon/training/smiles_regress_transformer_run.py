from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence, text
import logging
from .ST_funcs.smiles_regress_transformer_funcs import train_val_data, assemble_callbacks
from data_loader.model_loader import retrieve_model_from_dict, save_model_weights
import sys
import os
from time import perf_counter
import time
import intel_extension_for_tensorflow as itex
import dragon
from dragon.data.ddict.ddict import DDict
from logging_config import train_logger as logger
from logging_config import stdout_to_logger, driver_logger

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


    tic_start = perf_counter()

    prev_top_candidates = []
    training_iter = 0
    while continue_training(continue_event, training_iter):

        try:
            current_sort_list = model_list_dd.bget("current_sort_list")
            top_candidates = current_sort_list['smiles']
            simulated_compounds = model_list_dd.bget("simulated_compounds")
        except:
            top_candidates = []

        logger.info(f"current_sort_list has {len(top_candidates)} candidates")
        
        if top_candidates == prev_top_candidates or len(simulated_compounds) < len(top_candidates):
            # Sleep if the top candidates have not changed or there are too few simulations
            logger.info("No new candidates to train on, waiting...")
            time.sleep(list_poll_interval_sec)
        else:
            ######## Build model #############
            if len(simulated_compounds) < len(top_candidates):
                logger.info("Too few simulations to train, skipping this iteration")
                logger.info(f"Number of simulated compounds: {len(simulated_compounds)}")
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
            
            logger.info(f"There are {len(x_train)} simulations to train on")
            # Only train if there is new data
            if len(x_train) > 0:
                prev_top_candidates = top_candidates.copy()
                logger.info("Starting Training")
                logger.info(f"{BATCH=} {EPOCH=} {len(x_train)=}")
                
                with stdout_to_logger(logger, level=logging.INFO):
                    fit_tic = perf_counter()
                    history = model.fit(
                                x_train,
                                y_train,
                                batch_size=BATCH,
                                epochs=EPOCH,
                                verbose=2,
                                validation_data=(x_val,y_val),
                                callbacks=callbacks,
                            )
                    fit_toc = perf_counter()
                    #print("model fitting complete")
                #sys.stdout = sys.__stdout__
                logger.info(f"model fitting complete in {fit_toc-fit_tic} seconds")
                
                # Save to dictionary
                # Retrieve simulated compounds to copy back after checkpointing
                simulated_compounds = model_list_dd.bget("simulated_compounds")
                # Checkpoint model/list dictionary
                model_list_dd.checkpoint()
                model_iter = model_list_dd.checkpoint_id
                logger.info(f"Model/List dictionary moved to checkpoint {model_iter}")
                # Restore simulated compounds
                model_list_dd.bput("simulated_compounds", simulated_compounds)
                # Save model weights to dictionary
                save_model_weights(model_list_dd, model)
                logger.info(f"Saved model weights to dictionary {model_iter=}")
                
                # Write out model to file if desired
                if save_model:
                    model_path = "current_model.keras"
                    model.save(model_path)
                    logger.info(f"Saving model: {model_iter=} {model_path=}")
                logger.info("Saved fine tuned model to dictionary")

                # Notify other processes of new model and wait for them to reach barrier
                if new_model_event is not None:
                    new_model_event.set()
                    driver_logger.info(f"Training setting new_model_event and waiting for barrier to advance to checkpoint")
                    barrier.wait()
                    driver_logger.info(f"Barrier passed, clearing new_model_event, progressing with model {model_iter}")
                    new_model_event.clear()
        training_iter += 1
            


