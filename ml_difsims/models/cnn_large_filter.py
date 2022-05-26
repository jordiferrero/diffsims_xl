#%%
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
import os, glob
import matplotlib.pyplot as plt
import pandas as pd
import random
import h5py
from mongodb.pymongo_connect import connect_to_mongo_database

#%%
tf.config.list_physical_devices()

#%% md
## Set up parameter variables
#%%

# Randomisation
seed = 10
random.seed(seed)
np.random.seed(seed)
# Unique identifier
import uuid
unique_id = uuid.uuid4()

model_name = f"model-{unique_id}"

# Connect your script to Neptune
import neptune.new as neptune
from neptune.new.types import File
from secrets import neptune_api_token, neptune_project

run = neptune.init(api_token=neptune_api_token,
                   project=neptune_project)

json_vars = {
    "root_path" : r'G:\My Drive\PhD\projects\external_measurements\ml_difsims',
    "cnn_to_use" : "large_filter_cnn",
    # Select between 'large_filter_cnn' or 'binary_then_multiclass'
    "cnn_parameters" : {
        "epochs" : 2,
    },
    "train_test_datasets" : {
        "batch_size" : 64,
        "test_size_percentage" : 0.25,
        "cropping_type" : "q",
        # Select cropping type: either q, px or all.
        "sim_unique_id" : "sim-94b798f2-69ac-4a64-b436-33f2af3bf104.hdf5",
        "files_location_from_root" : r"data/simulations",
    },
    "random_seed" : seed,
    "id": model_name,
    "neptune_run_url" : str(run.get_run_url()),
    "save_model_relpath" : r'data/models',
}

#%%
import json
from types import SimpleNamespace

json_vars_dump = json.dumps(json_vars)
vs = json.loads(json_vars_dump, object_hook=lambda d: SimpleNamespace(**d))

#%%
epochs = vs.cnn_parameters.epochs
batch_size = vs.train_test_datasets.batch_size
cnn_to_use = vs.cnn_to_use

# Load simulation data
root = vs.root_path
sim_data_folder = os.path.join(root, vs.train_test_datasets.files_location_from_root)
sim_fname = vs.train_test_datasets.sim_unique_id
sim_data_path = os.path.join(sim_data_folder, sim_fname)

#%% md
## Load data
with h5py.File(sim_data_path, 'r') as f:
    sim_data = f[f'x_{vs.train_test_datasets.cropping_type}']
    sim_data = np.array(sim_data)
    sim_labels = f[f'y_{vs.train_test_datasets.cropping_type}']
    sim_labels = np.array(sim_labels)
    phase_names = f['metadata'].attrs['phases']
    json_sim_metadata = f['metadata']['metadata_json']

phase_names = [s for s in phase_names]

# Reshape to just (data, label)
sim_data = sim_data.reshape(-1, sim_data.shape[-1])

# Create labels (read them from file instead)
n_phases = len(phase_names)
sim_labels = np.zeros((n_phases, int(sim_data.shape[0]/n_phases)))
for i in range(n_phases):
    sim_labels[i,:] = i
sim_labels = sim_labels.flatten()
print(phase_names)
#%% md
## Create training and testing datasets
#%%
train_data, test_data, train_labels, test_labels = train_test_split(sim_data, sim_labels,
                                                                    test_size=vs.train_test_datasets.test_size_percentage,
                                                                    random_state=seed)
print('train_data & train_labels:')
print(np.shape(train_data), np.shape(train_labels))
print('test_data & test_labels:')
print(np.shape(test_data), np.shape(test_labels))

#%% md
## Connect to `neptune.io` and start experiment
#%%
PARAMS = {
'n_classes': len(phase_names),
'epochs': epochs,
'batch_size': batch_size,
'cnn_to_use': cnn_to_use,
}

tags = [cnn_to_use] + phase_names

run["sys/tags"].add(tags)
run['source_code/model/my_params'].log(PARAMS)
run['source_code/simulation_dataset_name'].log(os.path.basename(sim_data_path))
run['source_code/model/unique_id'].log(model_name)

#%% md
## Create Neural Network
#%%
# Reshape to create categorical labels (instead of value from 0-n, get an n-array with 0s and 1)
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#%%
def create_large_filter_model(input_shape, output_classes):
    conv1d_filters = 64
    conv1d_kernel_size = 6
    deep_layer_input = 128
    max_pooling = 2
    dropout_rate = 0.5
    model = Sequential()
    model.add(tf.keras.layers.Conv1D(conv1d_filters, conv1d_kernel_size,
                                     input_shape=input_shape,
                                     data_format='channels_last',
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(max_pooling))
    model.add(tf.keras.layers.Conv1D(conv1d_filters, conv1d_kernel_size,
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(max_pooling))
    model.add(Dropout(dropout_rate))
    model.add(tf.keras.layers.Conv1D(conv1d_filters, conv1d_kernel_size,
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(max_pooling))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(deep_layer_input, activation = 'relu'))
    model.add(tf.keras.layers.Dense(output_classes,  activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer = 'adam',
                  metrics=['accuracy'])
    return model


def binary_then_multiclass(input_shape, output_classes):
    model = Sequential()
    return model
#%%
input_shape = (train_data[0].size, 1)
output_classes = len(phase_names)

if cnn_to_use == 'large_filter_cnn':
    model = create_large_filter_model(input_shape, output_classes)
elif cnn_to_use == 'binary_then_multiclass':
    raise NotImplementedError("To be added")
else:
    raise NotImplementedError("The model asked is not implemented yet.")
model.summary()
#%%
batch_size = PARAMS['batch_size']
epochs =  PARAMS['epochs']

from neptune.new.integrations.tensorflow_keras import NeptuneCallback
neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')

history = model.fit(train_data, train_labels,
          batch_size=batch_size, epochs=epochs,
          callbacks=[neptune_cbk])
#%%
accuracy_train = model.evaluate(train_data,train_labels,)
accuracy_test = model.evaluate(test_data, test_labels,)
#%%
data = np.vstack([accuracy_train, accuracy_test])
accuracies = pd.DataFrame(data, index=['train', 'test'], columns=['loss', 'accuracy'])
run['metrics/train/accuracies_df'].upload(File.as_html(accuracies))
accuracies


#%%
import datetime
now = datetime.datetime.now()
timestamp = now.strftime("%Y%m%d")

train_ac_str = f"{accuracy_train[1]:.4f}".replace(".", "p")
test_ac_str = f"{accuracy_test[1]:.4f}".replace(".", "p")

model_description = f'{timestamp}_CNN_{cnn_to_use}_{output_classes}nclasses_' \
       f'{epochs}epochs_{batch_size}batchsize__train_{train_data.shape[0]/output_classes:.0f}n_' \
       f'{train_ac_str}ac__test_{test_data.shape[0]/output_classes:.0f}n_' \
       f'{accuracy_test[1]:.4f}ac'

#%%
fpath = os.path.join(root, vs.save_model_relpath, f'{model_name}.hdf5')
model.save(fpath)

with h5py.File(fpath, 'w') as f:
    g = f.create_group('metadata')
    g.create_dataset('metadata_json', data=json_vars_dump)
    g.attrs['phases'] = phase_names
    # TODO: Fix the line below
    #g.create_dataset('metadata_from_simulation_json', data=json_sim_metadata)


run['source_code/model/model_name'].log(fpath)
run['source_code/model/model_description'].log(model_description)
run['source_code/model/phase_names'].log(phase_names)
model.summary(print_fn=lambda x: run['source_code/model/model_summary'].log(x))
model.summary()

# Send json database to mongodb
db_collection = connect_to_mongo_database('models', model_name)
db_collection.insert_one(json_vars)

#%% md
## Undestanding misclassification in test data
#%%
# Get the metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pandas as pd

preds = np.round(model.predict(test_data), 0)
classification_metrics = metrics.classification_report(test_labels, preds, target_names=phase_names, output_dict=True)
classification_metrics = pd.DataFrame(classification_metrics).T

cf_matrix = confusion_matrix(test_labels.argmax(1), preds.argmax(1),)
#%%
classification_metrics.iloc[:-4,:-1].plot()
fig = plt.gcf()
run['metrics/test/classification_metrics'].upload(File.as_html(fig))
plt.close()
#%%
run['metrics/test/classification_metrics'].upload(File.as_html(classification_metrics))
print(classification_metrics)
#%%
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    import itertools
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt.gcf()
#%%
fig = plot_confusion_matrix(cf_matrix, phase_names, normalize=False)
run['metrics/test/confusion_matrix'].upload(File.as_image(fig))
plt.close()
#%%
fig_norm = plot_confusion_matrix(cf_matrix, phase_names, normalize=True)
run['metrics/test/confusion_matrix_norm'].upload(File.as_image(fig_norm))
plt.close()
#%%

# Plot some of the misclassified data:
bool_predictions = test_labels.argmax(1) == preds.argmax(1)

n_bad = np.count_nonzero(~bool_predictions)
n_max = min(25, n_bad)

fig, axs = plt.subplots(nrows=n_max, figsize = (10, n_max * 1.5), sharex=True)
ax = 0
for i, bool_pred in enumerate(bool_predictions):
    if ax >= n_max:
        break
    if bool_pred == False:
        true_phase = phase_names[test_labels.argmax(1)[i]]
        pred_phase = phase_names[preds.argmax(1)[i]]
        lab = 'True: {}, Pred: {}'.format(true_phase, pred_phase)
        axs[ax].plot(test_data[i,:], label=lab, color=f'C{ax}')
        axs[ax].legend()
        ax += 1

run['metrics/test/bad_predictions'].upload(File.as_image(fig))
plt.close()
#%%
run.stop()
#%%
