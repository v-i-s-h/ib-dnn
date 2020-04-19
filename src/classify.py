"""
    Script for training classification models
"""

import os, datetime, json

import tensorflow as tf
import larq as lq

import click
from zookeeper import cli, build_train

from experiment import Experiment
import models, data
import callbacks
import utils

# To avoid error from K.function([..., K.learning_phase()], [...])
tf.compat.v1.disable_eager_execution()

# Register train command and associated swicthes
@cli.command()
@click.option("--name", default="classify")
@click.option("--observer", default=None)
@build_train()
def train(build_model,      # build function from `models`
          dataset,          # dataset from `data`
          hparams,          # hyper parameters from `models`
          logdir,           # log directory to save results
          name,             # name of the experiment - for sacred
          observer):        # sacred observer for the experiment

    # Check if the given directory already contains model
    if os.path.exists(f"{logdir}/stats.json"):
        # then we will load the model weights
        model_dir = logdir
    else:
        # otherwise, create 
        # location to save the model -- <logdir>/<name>/<dataset>/<model>/<timestamp>
        model_dir = os.path.join(logdir,
                                name,
                                dataset.dataset_name,
                                build_model.__name__,
                                datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "weights.h5")
    
    # check if any observers are added for experiment.
    # if not, add a file_storage observer
    if observer is None:
        observer = f"file_storage={model_dir}"

    # Create an experiment
    ex = Experiment(name, 
                    dataset, 
                    build_model.__name__, 
                    hparams, 
                    observer)

    # Main function to run experiment
    @ex.main
    def train(_run):
        # build model
        model = build_model(hparams, **dataset.preprocessing.kwargs)

        # compile model
        model.compile(
            optimizer=utils.make_optimizer(hparams.optimizer, 
                                                hparams.opt_param),
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"]
        )

        # print summary of created model
        lq.models.summary(model)

        # if model already exists, load it and continue training
        initial_epoch = 0
        if os.path.exists(os.path.join(model_dir, "stats.json")):
            with open(os.path.join(model_dir, "stats.json"), "r") as stats_file:
                initial_epoch = json.load(stats_file)["epoch"]
                click.echo(f"Restoring model from {model_path} at epoch = {initial_epoch}")
                model.load_weights(model_path)

        # attach callbacks
        # save model at the end of each epoch
        training_callbacks = [
            callbacks.SaveStats(model_dir=model_dir)
        ]
        # send data to sacred experiment
        training_callbacks.extend([
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs:
                    [
                        ex.log_scalar(metric, value, epoch+1) for (metric, value) in logs.items()
                    ]
            )
        ])
        # compute MI
        mi_estimator = callbacks.EstimateMI(dataset.load_split("test"),
                                            monitor_layers = {
                                                "quant_dense": lq.layers.QuantDense,
                                                "batchnorm": tf.keras.layers.BatchNormalization,
                                                "activation": tf.keras.layers.Activation
                                            },
                                            log_file=os.path.join(model_dir, "mi_data.json")
                                        )
        training_callbacks.extend([mi_estimator])
        # custom prgress bar
        training_callbacks.extend([callbacks.ProgressBar(initial_epoch)])
        
        # train the model
        train_log = model.fit(
                            dataset.train_data(hparams.batch_size),
                            epochs=hparams.epochs,
                            steps_per_epoch=dataset.train_examples // hparams.batch_size,
                            validation_data=dataset.validation_data(hparams.batch_size),
                            validation_steps=dataset.validation_examples // hparams.batch_size,
                            initial_epoch=initial_epoch,
                            callbacks=training_callbacks,
                            verbose=0
        )

    # # Execute experiment
    ex.execute()

if __name__ == "__main__":
    cli()