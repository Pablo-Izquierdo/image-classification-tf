import os
import json
import numpy as np
import matplotlib.pylab as plt

from tensorflow.keras.models import load_model

from imgclas.data_utils import load_image, load_data_splits
from imgclas.test_utils import predict
from imgclas import paths, plot_utils, utils


# User parameters to set
TIMESTAMP = '2024-02-01_115602'                       # timestamp of the model
MODEL_NAME = 'epoch-20.hdf5'


try:
    os.mkdir(f'/srv/image-classification-tf/models/{TIMESTAMP}/predictions/bootstrapping-epoch-20/')
except Exception as e:
    print("directory exits")

for sample in os.listdir('/srv/image-classification-tf/models/2024-02-01_115602/dataset_files/bootstrapping'): 
   
    print(f'Executing... {sample}')
    # Set the timestamp
    paths.timestamp = TIMESTAMP

    # Load training configuration
    conf_path = os.path.join(paths.get_conf_dir(), 'conf.json')
    with open(conf_path) as f:
        conf = json.load(f)

    filepath = os.path.join(paths.get_checkpoints_dir(), MODEL_NAME) 
    obj=utils.get_custom_objects()

    # Load the model
    model = load_model(filepath, custom_objects=obj, compile=False)

    
    
    SPLIT_NAME = sample.split('.')[0]                         # data split to use 
    # Load the data
    X, y = load_data_splits(splits_dir=paths.get_ts_splits_dir(),
                            im_dir=conf['general']['images_directory'],
                            split_name=f'bootstrapping/'+SPLIT_NAME)

    # Predict
    pred_result = predict(model, X, conf, filemode='local')

    # Save the predictions
    pred_dict = {'filenames': list(X),
                 'pred_value': pred_result.tolist()}
    if y is not None:
        pred_dict['true_value'] = y.tolist()

    pred_path = os.path.join(paths.get_predictions_dir()+f'/bootstrapping-epoch-20/', '{}.json'.format(SPLIT_NAME))
    with open(pred_path, 'w') as outfile:
        json.dump(pred_dict, outfile, sort_keys=True)