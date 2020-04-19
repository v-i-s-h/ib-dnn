# MNIST data and related datasets

from zookeeper import registry
from .tasks import ImageClassification

@registry.register_preprocess("mnist")
class default(ImageClassification):
    def inputs(self, data):
        return super().inputs(data) / 255.0

@registry.register_preprocess("mnist")
class zerocenter(ImageClassification):
    def inputs(self, data):
        return super().inputs(data) / 127.5 - 1.0