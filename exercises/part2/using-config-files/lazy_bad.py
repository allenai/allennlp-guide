import json

from allennlp.common import FromParams, Params, Lazy
from allennlp.data import Vocabulary


class Gaussian(FromParams):
    def __init__(self, vocab: Vocabulary, mean: float, variance: float):
        self.vocab = vocab
        self.mean = mean
        self.variance = variance
        print(f"Gaussian got vocab with object id: {id(vocab)}")


class ModelWithGaussian(FromParams):
    def __init__(self, gaussian: Lazy[Gaussian]):
        # Pretend that we needed to do some non-trivial processing / reading from
        # disk in order to construct this object.
        self.vocab = Vocabulary()
        self.gaussian = gaussian.construct(vocab=self.vocab)


param_str = """{"gaussian": {"mean": 0.0, "variance": 1.0}}"""
params = Params(json.loads(param_str))

model = ModelWithGaussian.from_params(params=params)
print("Mean:", model.gaussian.mean)
print("Variance:", model.gaussian.variance)
