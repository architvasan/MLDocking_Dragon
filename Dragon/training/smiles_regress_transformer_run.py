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
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence, text
import pandas as pd

#from .ST_funcs.clr_callback import *
from .ST_funcs.smiles_regress_transformer_funcs import train_val_data, assemble_callbacks, train_val_data_archit
from data_loader.model_loader import retrieve_model_from_dict, save_model_weights
import sys
import os

import dragon
from dragon.data.ddict.ddict import DDict

#tf.config.run_functions_eagerly(True)
#tf.enable_eager_execution()

class LearningRatePrinter(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        # In case lr is a schedule, evaluate it
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr_value = lr(self.model.optimizer.iterations)
        else:
            lr_value = lr
        print(f"Epoch {epoch+1}: Learning rate is {tf.keras.backend.get_value(lr_value):.6f}")

class LossCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        # logs contain the loss and any other metrics
        print(f"\nBatch {batch} - Loss: {logs.get('loss'):.4f}")

class DebugCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        weights = self.model.trainable_weights
        print(f"\nBatch {batch}:")
        for w in weights:
            print(f"{w.name} mean={tf.reduce_mean(w).numpy():.6f}")


def fine_tune(dd: DDict, 
                candidate_dict: DDict, 
                BATCH: int, 
                EPOCH: int, 
                save_model=True):

    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    tf.keras.mixed_precision.set_global_policy('float32')

    fine_tune_log = "training.log"

    ######## Build model #############
    model, model_iter, hyper_params = retrieve_model_from_dict(dd,fine_tune=True)
    model_iter += 1
    with open(fine_tune_log, 'w') as f:
        f.write(f"Read model weights from DDict\n")

    ######## Create training and validation data##### 
    with open(fine_tune_log, 'a') as f:
        f.write(f"Create training data\n")
    x_train, y_train, x_val, y_val = train_val_data(candidate_dict,method="stratified")    
    #x_train, y_train, x_val, y_val = train_val_data_archit('/flare/hpe_dragon_collab/balin/archit_fine_tune')    
    with open(fine_tune_log, 'a') as f:
        f.write(f"Finished creating training data\n")
        f.write(f'Max docking {y_train.max()},{y_val.max()}\n')
        f.write(f'Min docking {y_train.min()},{y_val.min()}\n')
 
    ######## Create callbacks #######
    callbacks = assemble_callbacks(hyper_params)
 
    ######## Train #######
    # Only train if there is new data
    if len(x_train) > 0:
        with open(fine_tune_log, 'a') as f:
            f.write(f"{BATCH=} {EPOCH=} {len(x_train)=}\n")
            #f.write("\n\nTRAINING DATA\n")
            #for n in range(64):
            #    f.write(f"{x_train[n]}  {y_train[n]}\n")
            #f.write("\n\nVALIDATION DATA\n")
            #for n in range(64):
            #    f.write(f"{x_val[n]}  {y_val[n]}\n")
        
        with open(fine_tune_log, 'a') as sys.stdout:
            #output = model.predict(x_train[:64], batch_size=64, verbose=0).flatten()
            #output = model(x_train[:64], training=True).numpy().flatten()
            #print('predictions: ',output)
            #print('truth: ',y_train[:64].flatten())
            #print('mse Train=True (should be same as model.fit() loss): ',np.mean(np.square(output - y_train[:64].flatten())))
            #output = model(x_train[:64], training=False).numpy().flatten()
            #print('predictions: ',output)
            #print('mse Train=False (same as model.predict()): ',np.mean(np.square(output - y_train[:64].flatten())))
            history = model.fit(
                        x_train,
                        y_train,
                        batch_size=BATCH,
                        epochs=EPOCH,
                        verbose=2,
                        validation_data=(x_val,y_val),
                        shuffle=True,
                        callbacks=callbacks,
                    )
            #print("model fitting complete",flush=True)
            #output = model.predict(x_train[:64], batch_size=64, verbose=0).flatten()
            #print('predictions: ',output)
            #print('mse: ',np.mean(np.square(output - y_train[:64].flatten())))
        sys.stdout = sys.__stdout__
        print("model fitting complete",flush=True)
        
        
        # Save to dictionary
        if save_model:
            model_path = "current_model.keras"
            model.save(model_path)
            with open("model_iter",'w') as f:
                f.write(f"{model_iter=} {model_path=}")

        save_model_weights(dd, model, model_iter)
        print("Saved fine tuned model to dictionary",flush=True)
    


