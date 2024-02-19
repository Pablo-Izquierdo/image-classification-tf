
"""
Miscellaneous functions manage data.

Date: November 2021
Authors: Miriam Cobo, Ignacio Heredia
Email: cobocano@ifca.unican.es, iheredia@ifca.unican.es
Github: miriammmc, ignacioheredia
"""

import os
import threading
from multiprocessing import Pool
import queue
import subprocess
import warnings
import base64

import numpy as np
import requests
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical, Sequence
import cv2
import albumentations
from albumentations.augmentations import transforms
from albumentations.imgaug import transforms as imgaug_transforms
import paths
from imgclas.data_utils import load_image, load_data_splits
from imgclas.test_utils import predict
from imgclas import paths, plot_utils, utils




def split_data_cross_validation(splits_dir, im_dir, split_name='dataset'):
    
    X, y = load_data_splits(splits_dir=splits_dir,
                     im_dir=im_dir,
                     split_name=split_name)
    
    imagenes = list()
    label = list()
    #print(X)
    for d,l in zip(X, y): # Para cada directorio
        for img in os.listdir(d): # Para cada imagen
            path = d + img
            print(path)
            # Load images
            data = load_image(path)
            imagenes.append(data)
            label.append(l)
    
    imagenes = np.array(imagenes)
    label = np.array(label)
    return X, y

class cross_val_model:
    def __init__(self, model):
        self.model = model
        
    def get_params(self, deep=True):
    # suppose this estimator has parameters "alpha" and "recursive"
        return {"model": self.model}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def predict(self, X, y):
        
        '''
        X : str or list
            List of images paths of length N. If providing a list of urls, be sure to set correctly the 'filemode' parameter.
            If a str is provided it will be understood as a single image to predict.
        conf: dict
            Configuration parameters. The data augmentation parameters that will be used in the inference can be changed in
            conf['augmentation']['val_mode'].
        filemode : str, {'local','url'}
            - 'local': filename is absolute path in local disk.
            - 'url': filename is internet url.
        '''
        #TODO: revisar como hacer predict para ver cual es mejor
        # Predict
        pred_result = predict(self.model, X, conf, filemode='local')

        # Save the predictions
        pred_dict = {'filenames': list(X),
                     'pred_value': pred_result.tolist()}
        if y is not None:
            pred_dict['true_value'] = y.tolist()

        pred_path = os.path.join(paths.get_predictions_dir(), '{}+{}.json'.format(MODEL_NAME, SPLIT_NAME))
        with open(pred_path, 'w') as outfile:
            json.dump(pred_dict, outfile, sort_keys=True)
        
        return 0
    
    '''
    def classify(self, inputs):
        return sign(self.predict(inputs))
    '''

    def fit(self, X, y, **kwargs):
        print(len(kwargs))
        print(kwargs)
        
        # CREAR GENERATOR
        train_gen = data_sequence(X, y,
                              batch_size=kwargs['CONF']['training']['batch_size'],
                              im_size=kwargs['CONF']['model']['image_size'],
                              mean_RGB=kwargs['CONF']['dataset']['mean_RGB'],
                              std_RGB=kwargs['CONF']['dataset']['std_RGB'],
                              preprocess_mode=kwargs['CONF']['model']['preprocess_mode'],
                              aug_params=kwargs['CONF']['augmentation']['train_mode'])
        train_steps = int((np.ceil(len(X_train)/kwargs['CONF']['training']['batch_size'])))
        
        # ENTRENAMIENTO
        history = self.model.fit_generator(generator=train_gen,
                                  steps_per_epoch=train_steps,
                                  epochs=kwargs['CONF']['training']['epochs'],
                                  validation_data=kwargs["validation_data"],
                                  validation_steps=kwargs["validation_steps"],
                                  callbacks=kwargs["callbacks"],
                                  verbose=kwargs["verbose"], 
                                  max_queue_size=kwargs["max_queue_size"],
                                  workers=kwargs["workers"],
                                  use_multiprocessing=kwargs['CONF']['training']['use_multiprocessing'],
                                  initial_epoch=kwargs["initial_epoch"])
        
        # Saving everything
        print('Saving data to {} folder.'.format(paths.get_timestamped_dir()))
        print('Saving training stats ...')
        stats = {'epoch': history.epoch,
                 'training time (s)': round(time.time()-t0, 2),
                 'timestamp': kwargs['TIMESTAMP'],
                 'mean RGB pixel': kwargs['CONF']['dataset']['mean_RGB'],
                 'standard deviation of RGB pixel': kwargs['CONF']['dataset']['std_RGB'],
                 'batch_size': kwargs['CONF']['training']['batch_size']}
        stats.update(history.history)
        stats = json_friendly(stats)
        stats_dir = paths.get_stats_dir()
        with open(os.path.join(stats_dir, 'stats.json'), 'w') as outfile:
            json.dump(stats, outfile, sort_keys=True, indent=4)

        print('Saving the configuration ...')
        model_utils.save_conf(kwargs['CONF'])

        print('Saving the model to h5...')
        fpath = os.path.join(paths.get_checkpoints_dir(), 'final_model.h5')
        self.model.save(fpath,
                   include_optimizer=True)
        
        
        return history
        
    '''def get_params(self, deep = False):
        return {'l':self.l}
    '''

#split_data_cross_validation("/srv/image-classification-tf/models/2023-09-14_105236/dataset_files/", paths.get_images_dir())