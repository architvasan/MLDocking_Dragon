from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence, text
import logging
from .ST_funcs.smiles_regress_transformer_funcs import train_val_data, assemble_callbacks
from data_loader.model_loader import retrieve_model_from_dict, save_model_weights
import sys
import os
from time import perf_counter
import time
try:
    import intel_extension_for_tensorflow as itex
except ImportError:
    pass
import dragon
from dragon.data.ddict.ddict import DDict
from logging_config import train_logger as logger
from logging_config import stdout_to_logger, driver_logger

def continue_training(stop_event, training_iter, model_iter, max_iter):
    """Determine whether to continue training based on event and iteration count"""
    sequential_workflow = stop_event is None
    if sequential_workflow:
        if training_iter == 0:
            return True
        else:
            return False
    else:
        if max_iter is not None:
            if model_iter >= max_iter:
                stop_event.set()
                driver_logger.info(f"Reached maximum training iterations {max_iter}, stopping async workflow")
        return not stop_event.is_set()

def fine_tune(model_list_dd: DDict, 
                sim_dd: DDict,
                stop_event,
                new_model_event,
                barrier,
                BATCH: int, 
                EPOCH: int,
                max_iter=None,
                save_model=True,
                list_poll_interval_sec=60):

    logger.info("Starting fine tune training")
    sequential_workflow = stop_event is None
    tic_start = perf_counter()
    prev_top_candidates = []
    training_iter = 0
    model_iter = model_list_dd.checkpoint_id
    while continue_training(stop_event,
                            training_iter, 
                            model_iter,
                            max_iter):
        # This will block until current sort list is available
        logger.info("Waiting for current_sort_list")
        current_sort_list = model_list_dd.bget("current_sort_list")
        top_candidates = current_sort_list['smiles']
        logger.info("Retrieved current_sort_list")
        logger.info(f"current_sort_list has {len(top_candidates)} candidates")
        # Once the sorted list is available clear the new model event
        logger.info(f"{sequential_workflow=}")
        if not sequential_workflow:
            logger.info(f"Status of new_model_event is {new_model_event.is_set()=}")
            if new_model_event.is_set():
                new_model_event.clear()
                driver_logger.info("Cleared new_model_event")
                logger.info("Cleared new_model_event")
        
        simulated_compounds = list(sim_dd.keys())
        model_list_dd.bput('simulated_compounds', simulated_compounds)
        #simulated_compounds = model_list_dd.bget("simulated_compounds")
        logger.info(f"Number of simulations available for training: {len(simulated_compounds)}")
        if top_candidates == prev_top_candidates:
            # Sleep if the top candidates have not changed
            logger.info("No new candidates to train on, waiting...")
            time.sleep(list_poll_interval_sec)
        elif len(simulated_compounds) < len(top_candidates)//2:
            #  Sleep if there are too few new simulations
            logger.info(f"Only {len(simulated_compounds)} simulations are available, waiting...")
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
                if not sequential_workflow:
                    new_model_event.set()
                    logger.info(f"Training setting new_model_event")
                    driver_logger.info(f"Training setting new_model_event")
        training_iter += 1
    if not sequential_workflow:
        new_model_event.set()
    logger.info("Training process exiting")
