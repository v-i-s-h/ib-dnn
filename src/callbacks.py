# Callbacks for training

import os, json
import numpy as np
from tqdm import tqdm 

import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_datasets as tfds

from mi_est import bin_compute_mi

class SaveStats(tf.keras.callbacks.Callback):
    """
        Callback for saving current model parameters and stats
    """
    def __init__(self, model_dir):
        super(SaveStats, self).__init__()

        self.model_dir = model_dir
        self.stats_file = os.path.join(model_dir, "stats.json")
        self.weights_file = os.path.join(model_dir, "weights.h5")
        self.metrics_file = os.path.join(model_dir, "metrics.json")

    def on_train_begin(self, logs=None):
        # build dictionary to log metrics
        self.metrics = {}
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, "r") as f:
                self.metrics = json.load(f)
        else:
            self.metrics['epoch'] = []
            for metric in self.params['metrics']:
                self.metrics[metric] = []
                
    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(self.weights_file)
        with open(self.stats_file, "w") as f:
            json.dump({"epoch": epoch + 1}, f)
        
        self.metrics['epoch'].append(epoch)
        for (metric, value) in logs.items():
            self.metrics[metric].append(value.astype(float))
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics, f)


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

        self.dataset = dataset
        
        # load y-data for the computation for conditional entropy in MI
        # test set
        testset = self.dataset.load_split("test", shuffle=False)
        y_test = np.stack([sample["label"] for sample in tfds.as_numpy(testset)]) 
        self.test_label_idx = dict()
        for i in range(self.dataset.info.features["label"].num_classes):
            self.test_label_idx[i] = y_test == i
        # train set
        trainset = self.dataset.load_split("train", shuffle=False)
        y_train = np.stack([sample["label"] for sample in tfds.as_numpy(trainset)]) 
        self.train_label_idx = dict()
        for i in range(self.dataset.info.features["label"].num_classes):
            self.train_label_idx[i] = y_train == i

        self.monitor_interval = 1
        self.monitor_layers = monitor_layers.copy()

        # dictionary to save MI values for each epoch and layer type
        self.mi_data = { 
                "train" :{layer_type: {} for (layer_type, _) in  self.monitor_layers.items()},
                "test"  :{layer_type: {} for (layer_type, _) in  self.monitor_layers.items()}
        }
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
                    if layer.name not in self.mi_data["train"][layer_type]: # if not reloaded
                        self.mi_data["train"][layer_type].update({layer.name: {}})
                        self.mi_data["test"][layer_type].update({layer.name: {}})

        # to build the layer wise activation model
        self.layer_out_names = {}
        self.layer_out_ops = []
        for (layer_type, act_ops) in self.layer_act_op.items():
            self.layer_out_names[layer_type] = list(act_ops.keys())
            self.layer_out_ops.extend(list(act_ops.values()))

        # Build Keras function to compute activations from each layer
        self.layer_act = tf.keras.models.Model(inputs=self.inp, outputs=self.layer_out_ops)
    
    def on_train_end(self, logs={}):
        self.update_mi_data(self.params['epochs'])
        if self.log_file:
            # save MI data
            with open(self.log_file, "w") as f:
                json.dump(self.mi_data, f)

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
            self.update_mi_data(epoch)

            if self.log_file:
                # save this epoch MI data
                # TODO: Incremental logging?
                with open(self.log_file, "w") as f:
                    json.dump(self.mi_data, f)

    def update_mi_data(self, epoch):
        # Compute activation from each layer on test data
        act = self.layer_act.predict(
                                        self.dataset.test_data(self.dataset.test_examples),
                                        steps=1
                                    )
        # pack it to layer_act_dict format
        layer_act_dict = {}
        count = 0
        for layer_type, layer_names in self.layer_out_names.items():
            l = len(layer_names)
            layer_act_dict[layer_type] = dict(zip(layer_names, act[count:count+l]))
            count += l
        for (layer_type, act_dict) in layer_act_dict.items():
            for layer_name, act_value in act_dict.items():
                mi_mx, mi_my = bin_compute_mi(self.test_label_idx, act_value, 0.5)
                self.mi_data["test"][layer_type][layer_name][epoch] = (mi_mx, mi_my)

        
        # Compute activation from each layer on train data
        act = self.layer_act.predict(
                                        self.extract_traindata(1000),
                                        steps=self.dataset.train_examples // 1000
                                    )
        # pack it to layer_act_dict format
        layer_act_dict = {}
        count = 0
        for layer_type, layer_names in self.layer_out_names.items():
            l = len(layer_names)
            layer_act_dict[layer_type] = dict(zip(layer_names, act[count:count+l]))
            count += l
        for (layer_type, act_dict) in layer_act_dict.items():
            for layer_name, act_value in act_dict.items():
                mi_mx, mi_my = bin_compute_mi(self.train_label_idx, act_value, 0.5)
                self.mi_data["train"][layer_type][layer_name][epoch] = (mi_mx, mi_my) 

    def extract_traindata(self, batch_size):
        # create train dataset object without shuffling
        train_dataset = self.dataset.maybe_cache(
            self.dataset.load_split(self.dataset.train_split, shuffle=False), "train_ordered"
        )
        return self.dataset._get_eval_data(train_dataset, batch_size)



# Custom progress bar
class ProgressBar(tf.keras.callbacks.Callback):
    """
        Custom Progress bar for monitoring training progress
    """
    def __init__(self, initial=0, metrics=None):
        """
        Arguments:
            initial     : epoch to which the progress bar needs to be set to.
            metrics     : metrics from `logs` to display
        """
        super(ProgressBar, self).__init__()
        self.initial = initial
        self.log_metrics = metrics

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.samples = self.params['samples']
        # create progress bar
        self.epochbar = tqdm(initial=self.initial, 
                             total=self.epochs, 
                             unit="epoch")
        if self.log_metrics is None:
            self.log_metrics = self.params['metrics']
        
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
        strings = ["{}: {:.3f}".format(metric, np.mean(logs[metric], axis=None)) for metric in self.log_metrics
                   if metric in logs]
        return " ".join(strings)