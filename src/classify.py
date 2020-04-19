"""
    Script for training classification models
"""

import os, datetime, json

import tensorflow as tf
import larq as lq

import click
from zookeeper import cli, build_train

import models, data
import callbacks
import utils

# To avoid error from K.function([..., K.learning_phase()], [...])
tf.compat.v1.disable_eager_execution()

# Register train command and associated swicthes
@cli.command()
@click.option("--name", default="classify")
@build_train()
def train(build_model,      # build function from `models`
          dataset,          # dataset from `data`
          hparams,          # hyper parameters from `models`
          logdir,           # log directory to save results
          name):            # name of the experiment

    # make configuration of this experiment
    config = {}
    for (param, value) in hparams.items():
        if not callable(value): # Filter out all non callable parameters
            if type(value) is dict:
                # If the hyper-parameter is specified as a dictionary,
                # then unpack it with proper names
                config[param] = {}
                for(_param, _value) in value.items():
                    if utils.is_jsonable(_value):
                        config[param][_param] = _value
                    else:
                        config[param][_param] = _value.__name__
            else:
                if utils.is_jsonable(value):
                    config[param] = value
                else:
                    config[param] = value.__name__
    config['dataset'] = dataset.dataset_name
    config['preprocess'] = type(dataset.preprocessing).__name__
    config['model'] = build_model.__name__    

    # Check if the given directory already contains model
    if os.path.exists(f"{logdir}/config.json"):
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

    # save current configuration
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    def train_model():
        # build model
        model = build_model(hparams, **dataset.preprocessing.kwargs)

        # compile model
        model.compile(
            optimizer=utils.make_optimizer(hparams.optimizer, 
                                                hparams.opt_param),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        # print summary of created model
        lq.models.summary(model)

        # if model already exists, load it and continue training
        initial_epoch = 0
        if os.path.exists(os.path.join(model_dir, "stats.json")):
            with open(os.path.join(model_dir, "stats.json"), "r") as stats_file:
                model_path = os.path.join(model_dir, "weights.h5")
                initial_epoch = json.load(stats_file)["epoch"]
                click.echo(f"Restoring model from {model_path} at epoch = {initial_epoch}")
                model.load_weights(model_path)

        # attach callbacks
        # save model at the end of each epoch
        training_callbacks = [
            callbacks.SaveStats(model_dir=model_dir)
        ]
        # compute MI
        mi_estimator = callbacks.EstimateMI(dataset.load_split("test"),
                                            hparams.mi_layer_types,
                                            log_file=os.path.join(model_dir, "mi_data.json")
                                        )
        training_callbacks.extend([mi_estimator])
        # custom prgress bar
        training_callbacks.extend([callbacks.ProgressBar(initial_epoch, ["accuracy"])])
        
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

        return train_log

    train_log = train_model()    

if __name__ == "__main__":
    cli()