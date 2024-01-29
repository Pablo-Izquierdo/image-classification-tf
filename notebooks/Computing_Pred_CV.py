import os
import json
import numpy as np
import matplotlib.pylab as plt

from tensorflow.keras.models import load_model

from imgclas.data_utils import load_image, load_data_splits
from imgclas.test_utils import predict
from imgclas import paths, plot_utils, utils


# User parameters to set
TIMESTAMP = '2024-01-24_085711'                       # timestamp of the model
for i in range(13,21):
    if i < 10:
        MODEL_NAME = f'epoch-0{i}.hdf5'
    else:
        MODEL_NAME = f'epoch-{i}.hdf5
    #MODEL_NAME = 'final_model.h5'                           # model to use to make the prediction
    print(f'Executing... {MODEL_NAME}')
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

    
    
    SPLIT_NAME = 'test'                          # data split to use 
    # Load the data
    X, y = load_data_splits(splits_dir=paths.get_ts_splits_dir(),
                            im_dir=conf['general']['images_directory'],
                            split_name=SPLIT_NAME)

    # Predict
    pred_result = predict(model, X, conf, filemode='local')

    # Save the predictions
    pred_dict = {'filenames': list(X),
                 'pred_value': pred_result.tolist()}
    if y is not None:
        pred_dict['true_value'] = y.tolist()

    pred_path = os.path.join(paths.get_predictions_dir(), '{}+{}.json'.format(MODEL_NAME, SPLIT_NAME))
    with open(pred_path, 'w') as outfile:
        json.dump(pred_dict, outfile, sort_keys=True)