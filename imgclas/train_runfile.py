"""
Training runfile

Date: November 2021
Authors: Miriam Cobo, Ignacio Heredia
Email: cobocano@ifca.unican.es, iheredia@ifca.unican.es
Github: miriammmc, ignacioheredia

Description:
This file contains the commands for training a convolutional net for image classification.

Additional notes:
* On the training routine: Preliminary tests show that using a custom lr multiplier for the lower layers yield to better
results than freezing them at the beginning and unfreezing them after a few epochs like it is suggested in the Keras
tutorials.
"""

#TODO List:

#TODO: Implement resuming training
#TODO: Try that everything works out with validation data
#TODO: Try several regularization parameters
#TODO: Add additional metrics for test time in addition to accuracy
#TODO: Implement additional techniques to deal with class imbalance (not only class_weigths)

import os
import time
import json
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from imgclas.data_utils import load_data_splits, compute_meanRGB, data_sequence, \
    json_friendly
from imgclas import paths, config, model_utils, utils
from imgclas.optimizers import customAdam
from imgclas.cross_validation import split_data_cross_validation, cross_val_model
from sklearn.model_selection import GroupKFold, cross_val_score

# Set Tensorflow verbosity logs
tf.logging.set_verbosity(tf.logging.ERROR)

# Dynamically grow the memory used on the GPU (https://github.com/keras-team/keras/issues/4161)
gpu_options = tf.GPUOptions(allow_growth=True)
tfconfig = tf.ConfigProto(gpu_options=gpu_options)
sess = tf.Session(config=tfconfig)
K.set_session(sess)


def train_fn(TIMESTAMP, CONF):
    
    print("GPU:", tf.test.is_gpu_available())

    paths.timestamp = TIMESTAMP
    paths.CONF = CONF

    utils.create_dir_tree()
    utils.backup_splits()

    # Load the training data
    X_train, y_train = load_data_splits(splits_dir=paths.get_ts_splits_dir(),
                                        im_dir=paths.get_images_dir(),
                                        split_name='train')

    # Load the validation data
    if (CONF['training']['use_validation']) and ('val.txt' in os.listdir(paths.get_ts_splits_dir())):
        X_val, y_val = load_data_splits(splits_dir=paths.get_ts_splits_dir(),
                                        im_dir=paths.get_images_dir(),
                                        split_name='val')
    else:
        print('No validation data.')
        X_val, y_val = None, None
        CONF['training']['use_validation'] = False

    # Load the class names
#     class_names = load_class_names(splits_dir=paths.get_ts_splits_dir()) ### for classification

    # Update the configuration
    CONF['model']['preprocess_mode'] = model_utils.model_modes[CONF['model']['modelname']]
    CONF['training']['batch_size'] = min(CONF['training']['batch_size'], len(X_train))

    # Compute the mean and std RGB values
    if CONF['dataset']['mean_RGB'] is None:
        CONF['dataset']['mean_RGB'], CONF['dataset']['std_RGB'] = compute_meanRGB(X_train)

    #Create data generator for train and val sets
    train_gen = data_sequence(X_train, y_train,
                              batch_size=CONF['training']['batch_size'],
#                               num_classes=CONF['model']['num_classes'], ###
                              im_size=CONF['model']['image_size'],
                              mean_RGB=CONF['dataset']['mean_RGB'],
                              std_RGB=CONF['dataset']['std_RGB'],
                              preprocess_mode=CONF['model']['preprocess_mode'],
                              aug_params=CONF['augmentation']['train_mode'])
    train_steps = int((np.ceil(len(X_train)/CONF['training']['batch_size'])))

    if CONF['training']['use_validation']:
        val_gen = data_sequence(X_val, y_val,
                                batch_size=CONF['training']['batch_size'],
#                                 num_classes=CONF['model']['num_classes'], ###
                                im_size=CONF['model']['image_size'],
                                mean_RGB=CONF['dataset']['mean_RGB'],
                                std_RGB=CONF['dataset']['std_RGB'],
                                preprocess_mode=CONF['model']['preprocess_mode'],
                                aug_params=CONF['augmentation']['val_mode'])
        val_steps = int((np.ceil(len(X_val)/CONF['training']['batch_size'])))
        
    else:
        val_gen = None
        val_steps = None

    # Launch the training
    t0 = time.time()

    # Create the model and compile it
    model, base_model = model_utils.create_model(CONF)

    # Get a list of the top layer variables that should not be applied a lr_multiplier
    base_vars = [var.name for var in base_model.trainable_variables]
    all_vars = [var.name for var in model.trainable_variables]
    top_vars = set(all_vars) - set(base_vars)
    top_vars = list(top_vars)

    # Set trainable layers
    if CONF['training']['mode'] == 'fast':
        for layer in base_model.layers:
            layer.trainable = False

    model.compile(optimizer=customAdam(lr=CONF['training']['initial_lr'],
                                       amsgrad=True,
                                       lr_mult=0.1,
                                       excluded_vars=top_vars
                                       ),
                  loss='mean_squared_error',
                  metrics=['mae']) ### for regression

    if not CONF['training']['use_cross_validation']:
        history = model.fit_generator(generator=train_gen,
                                  steps_per_epoch=train_steps,
                                  epochs=CONF['training']['epochs'],
                                  validation_data=val_gen,
                                  validation_steps=val_steps,
                                  callbacks=utils.get_callbacks(CONF),
                                  verbose=1, max_queue_size=5, workers=4,
                                  use_multiprocessing=CONF['training']['use_multiprocessing'],
                                  initial_epoch=0)
    
        # Saving everything
        print('Saving data to {} folder.'.format(paths.get_timestamped_dir()))
        print('Saving training stats ...')
        stats = {'epoch': history.epoch,
                 'training time (s)': round(time.time()-t0, 2),
                 'timestamp': TIMESTAMP,
                 'mean RGB pixel': CONF['dataset']['mean_RGB'],
                 'standard deviation of RGB pixel': CONF['dataset']['std_RGB'],
                 'batch_size': CONF['training']['batch_size']}
        stats.update(history.history)
        stats = json_friendly(stats)
        stats_dir = paths.get_stats_dir()
        with open(os.path.join(stats_dir, 'stats.json'), 'w') as outfile:
            json.dump(stats, outfile, sort_keys=True, indent=4)

        print('Saving the configuration ...')
        model_utils.save_conf(CONF)

        print('Saving the model to h5...')
        fpath = os.path.join(paths.get_checkpoints_dir(), 'final_model.h5')
        model.save(fpath,
                   include_optimizer=True)

        # print('Saving the model to protobuf...')
        # fpath = os.path.join(paths.get_checkpoints_dir(), 'final_model.proto')
        # model_utils.save_to_pb(model, fpath)
    
    else:
        # Cross-Validation
        X, y = split_data_cross_validation(splits_dir=paths.get_ts_splits_dir(),
                                        im_dir=paths.get_images_dir())
        
        kwargs = {
            "TIMESTAMP": TIMESTAMP,
            "CONF": CONF,
            "validation_data":None,
            "validation_steps":None,
            "callbacks":utils.get_callbacks(CONF),
            "verbose":1, 
            "max_queue_size":5,
            "workers":4,
            "initial_epoch":0}
        
        modelcv = cross_val_model(model)
        # Evaluate using 10-fold Cross Validation
        kfold = GroupKFold(n_splits=3)
        results = cross_val_score(modelcv, X, y, groups=y, cv=kfold, fit_params=kwargs, scoring = 'neg_mean_absolute_error')
        print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    

    print('Finished')


    
import yaml
import sys

if __name__ == '__main__':
    '''
        Arguments : Config file name.
        Not argument: takes config-r.yaml as default config file name
    '''
    n = len(sys.argv)
    if n > 1:
        if n > 2:
            raise Exception("Too many arguments, allowed only 1 arguments (config file name). or no arguments (config-r.yaml as default config file name)")

        homedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        conf_path = os.path.join(homedir, 'etc', str(sys.argv[1]))
        with open(conf_path, 'r') as f:
            CONF = yaml.safe_load(f)
            
        CONF = config.get_conf_dict(CONF)
    else:
        CONF = config.get_conf_dict()
        
    print(CONF)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    train_fn(TIMESTAMP=timestamp, CONF=CONF)
        
        
    #ORIGINAL EXECUTION
    #CONF = config.get_conf_dict(CONF)
    #timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    #train_fn(TIMESTAMP=timestamp, CONF=CONF)
