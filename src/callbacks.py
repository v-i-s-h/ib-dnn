# Callbacks for training

import os, json
import numpy as np
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
    def __init__(self, dataset, monitor_layers, *args, **kwargs):
        """
            Keras callback for computing mutual information
            Arguments:
                dataset     : TFDatasets split. Preferably test split.
                monitor_layers  : A dictionary of `"name" => type` format of layers to monitor
                    Eg:
                        monitor_layers = {
                            "quant_dense"   : lq.layers.QuantDense,
                            "batchnorm"     : tf.keras.layers.BatchNormalization,
                            "activations"   : tf.keras.layers.Activation
                        }
        """
        super(EstimateMI, self).__init__(*args, **kwargs)

        self.x_test = np.stack([sample["image"] for sample in tfds.as_numpy(dataset)])
        self.y_test = np.stack([sample["label"] for sample in tfds.as_numpy(dataset)])   

        print("EstimateMI")
        print(self.x_test.shape, self.y_test.shape)
        
        self.label_idx = dict()
        for i in range(10):         # NUM OF CLASSES
            self.label_idx[i] = self.y_test == i

        self.monitor_interval = 1
        self.monitor_layers = monitor_layers.copy()

        # dictionary to save MI values for each layer type and epoch
        self.mutual_information = { name: {} for (name, _) in self.monitor_layers.items()}


    def on_train_begin(self, logs={}):
        # Index all layers
        self.inp = self.model.input

        # Build dictionary of layers to monitor
        self.layer_out = dict()
        for name, layer_class in self.monitor_layers.items():
            self.layer_out[name] = [layer.output for layer in self.model.layers if isinstance(layer, layer_class)]
        
        # Build Keras function to compute activations from each layer
        self.layer_act = K.function([self.inp, K.learning_phase()], self.layer_out)

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
            print("Computing MI at epoch =", epoch)
            
            # Compute activation from each layer on test data
            layer_act_dict = self.layer_act([self.x_test, 1.0])

            for (name, act_value) in layer_act_dict.items():
                mi_list = dict()    # list to save mutual information
                for (layer_idx, _act) in enumerate(act_value):
                    # For each layer activation, compute MI I(M;X) and I(M;Y)
                    mi_mx, mi_my = bin_compute_mi(self.label_idx, _act, 0.5)
                    mi_list[layer_idx] = (mi_mx, mi_my)

                self.mutual_information[name][epoch] = mi_list