import json

from allennlp.common import FromParams, Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary


class Gaussian(FromParams):
    def __init__(self, vocab: Vocabulary, mean: float, variance: float):
        self.vocab = vocab
        self.mean = mean
        self.variance = variance
        print(f"Gaussian got vocab with object id: {id(vocab)}")


class ModelWithGaussian(FromParams):
    def __init__(self, vocab: Vocabulary, gaussian: Gaussian):
        self.vocab = vocab
        self.gaussian = gaussian
        print(f"ModelWithGaussian got vocab with object id: {id(vocab)}")

param_str = """{"gaussian": {"mean": 0.0, "variance": 1.0}}"""
params = Params(json.loads(param_str))
try:
    model = ModelWithGaussian.from_params(params)
except ConfigurationError:
    print("Caught ConfigurationError")

vocab = Vocabulary()

# Even though we're only passing `vocab=vocab` at the top level, the vocab object
# is available recursively to any objects that are constructed inside this call,
# including the Gaussian object.
model = ModelWithGaussian.from_params(params=params, vocab=vocab)
