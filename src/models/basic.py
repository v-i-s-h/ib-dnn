# Models for testing

import tensorflow as tf
from zookeeper import registry, HParams

# Fully connected network
@registry.register_model
def fcn(hparams, input_shape, num_classes):
    # 1. Create input layer
    input_layer = tf.keras.Input(shape=input_shape, name="input_layer")
    # 2. Flatten the input to make a vector
    x = tf.keras.layers.Flatten(name="flatten_layer")(input_layer)
    # 3. Create hidden layers
    for (h_idx, n_units) in enumerate(hparams.hidden_units):
        x = tf.keras.layers.Dense(n_units, name="lin_{:02d}".format(h_idx))(x)
        x = tf.keras.layers.Activation(hparams.activation, name="act_{:02d}".format(h_idx))(x)
    # 4. Output layer
    x = tf.keras.layers.Dense(num_classes, name="lin_out")(x)
    output_layer = tf.keras.layers.Activation("softmax", name="act_out")(x)

    # Return model object
    return tf.keras.models.Model(inputs=input_layer, 
                                 outputs=output_layer,
                                 name=hparams.model_name if hasattr(hparams, "model_name") else "fully_connected"
                                 )

                
@registry.register_hparams(fcn)
class default(HParams):
    # architecture
    activation = "tanh"
    hidden_units = [64, 64]

    # training
    epochs = 5
    batch_size = 64

    # Optimizer
    optimizer = "SGD"
    opt_param = dict(
        learning_rate = 0.001
    )

    # layers to monitor for MI
    mi_layer_types = {
        "activation": tf.keras.layers.Activation
    }

@registry.register_hparams(fcn)
class saxe2018iclr_mnist(default):
    """
        To reproduce results from 
        Saxe, Andrew M., et al. "On the information bottleneck theory of deep learning." ICLR 2018
    """
    hidden_units = [1024, 20, 20, 20]

    epochs = 10000
    batch_size = 64