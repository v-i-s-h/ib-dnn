import tensorflow as tf

# Utility function to make optimizer from hyper param settings
def make_optimizer(optimizer, opt_param):
    # Make optimizer from the given settings
    optimizer_lookup = {
        "Adam"      : tf.keras.optimizers.Adam,
        "SGD"       : tf.keras.optimizers.SGD,
        "RMSprop"   : tf.keras.optimizers.RMSprop
    }

    if optimizer in optimizer_lookup:
        # This is one of the traditional optimizers
        return optimizer_lookup[optimizer](**opt_param)
    else:
        raise RuntimeError(f"Optimizer '{optimizer} is not implemented.")