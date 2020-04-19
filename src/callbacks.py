# Callbacks for training

import os, json
import numpy as np
from tqdm import tqdm 

import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_datasets as tfds

from mi_est import bin_compute_mi

class SaveStats(tf.keras.callbacks.LambdaCallback):
    """
        Callback for saving current model parameters and stats
    """
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.stats_file = os.path.join(model_dir, "stats.json")
        self.weights_file = os.path.join(model_dir, "weights.h5")
        
        def save_stats(epoch, logs=None):
            self.model.save_weights(self.weights_file)
            with open(self.stats_file, "w") as f:
                json.dump({"epoch": epoch + 1}, f)
        
        super().__init__(on_epoch_end=save_stats)


class EstimateMI(tf.keras.callbacks.Callback):
    """
        Callback for computing mutual information I(T;X) and I(T;Y)
    """
    def __init__(self, dataset, monitor_layers, log_file=None, *args, **kwargs):
        """
            Keras callback for computing mutual information
            Arguments:
                dataset         : TFDatasets split. Preferably test split.
                monitor_layers  : A dictionary of `"name" => type` format of layers to monitor
                    Eg:
                        monitor_layers = {
                            "quant_dense"   : lq.layers.QuantDense,
                            "batchnorm"     : tf.keras.layers.BatchNormalization,
                            "activations"   : tf.keras.layers.Activation
                        }
                log_path        : Path of the file to log to.
        """
        super(EstimateMI, self).__init__(*args, **kwargs)

        # extract data from tfds format and make them numpy array for easy use with 
        # K.function(...) and mi comuputations
        self.x_test = np.stack([sample["image"] for sample in tfds.as_numpy(dataset)])
        self.y_test = np.stack([sample["label"] for sample in tfds.as_numpy(dataset)])   

        self.label_idx = dict()
        for i in range(10):         # NUM OF CLASSES
            self.label_idx[i] = self.y_test == i

        self.monitor_interval = 1
        self.monitor_layers = monitor_layers.copy()

        # dictionary to save MI values for each epoch and layer type
        self.mi_data = {layer_type: {} for (layer_type, _) in  self.monitor_layers.items()}
        # dictionary of operations in tf graph corresponding to the layer outputs, for K.function
        self.layer_act_op = {layer_type: {} for (layer_type, _) in self.monitor_layers.items()}

        # create a file stream to save the computations
        self.log_file = log_file
        if os.path.exists(self.log_file):
            with open(self.log_file, "r") as f:
                self.mi_data = json.load(f)

    def on_train_begin(self, logs={}):
        # Index all layers
        self.inp = self.model.input

        # Build dictionary of layers to monitor
        for layer_type, layer_class in self.monitor_layers.items():
            for layer in self.model.layers:
                if isinstance(layer, layer_class):
                    self.layer_act_op[layer_type].update({layer.name: layer.output})
                    if layer.name not in self.mi_data[layer_type]: # if not reloaded
                        self.mi_data[layer_type].update({layer.name: {}})

        # Build Keras function to compute activations from each layer
        self.layer_act = K.function([self.inp, K.learning_phase()], self.layer_act_op)
    
    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        # @TODO: Write a function for interval
        if epoch < 25:
            self.monitor_interval = 1
        elif epoch < 100:
            self.monitor_interval = 5
        elif epoch < 500:
            self.monitor_interval = 10
        elif epoch < 1000:
            self.monitor_interval = 50
        else:
            self.monitor_interval = 100

        if epoch % self.monitor_interval == 0:
            # Compute activation from each layer on test data
            layer_act_dict = self.layer_act([self.x_test, 0.0])

            for (layer_type, act_dict) in layer_act_dict.items():
                for layer_name, act_value in act_dict.items():
                    mi_mx, mi_my = bin_compute_mi(self.label_idx, act_value, 0.5)
                    self.mi_data[layer_type][layer_name][epoch] = (mi_mx, mi_my)

            if self.log_file:
                # save this epoch MI data
                # TODO: Incremental logging?
                with open(self.log_file, "w") as f:
                    json.dump(self.mi_data, f, indent=2)


# Custom progress bar
class ProgressBar(tf.keras.callbacks.Callback):
    def __init__(self, initial=0):
        super(ProgressBar, self).__init__()
        self.initial = initial

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.samples = self.params['samples']
        # create progress bar
        self.epochbar = tqdm(initial=self.initial, 
                             total=self.epochs, 
                             unit="epoch")
        
    def on_train_end(self, logs=None):
        self.epochbar.close()

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.epochbar.set_postfix_str(self.format_metrics(logs), refresh=False)
        self.epochbar.update(1)

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def format_metrics(self, logs):
        metrics = self.params['metrics']
        strings = ["{}: {:.3f}".format(metric, np.mean(logs[metric], axis=None)) for metric in metrics
                   if metric in logs]
        return " ".join(strings)