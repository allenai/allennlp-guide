import json

from allennlp.common import FromParams, Params, Registrable, Lazy
from allennlp.data import Vocabulary


class Gaussian(FromParams):
    def __init__(self, vocab: Vocabulary, mean: float, variance: float):
        self.vocab = vocab
        self.mean = mean
        self.variance = variance
        print(f"Gaussian got vocab with object id: {id(vocab)}")


class ModelWithGaussian(Registrable):
    def __init__(self, vocab: Vocabulary, gaussian: Gaussian):
        self.vocab = vocab
        self.gaussian = gaussian

    @classmethod
    def from_lazy_objects(cls, gaussian: Lazy[Gaussian]) -> "ModelWithGaussian":
        # Pretend that we needed to do some non-trivial processing / reading from
        # disk in order to construct this object.
        vocab = Vocabulary()
        gaussian_ = gaussian.construct(vocab=vocab)
        return cls(vocab=vocab, gaussian=gaussian_)


# In order to use a constructor other than __init__, we need to inherit from
# Registrable, not just FromParams, and register the class with the separate
# constructor.  And because we're registering the Registrable class itself, we
# can't do this as a decorator, like we typically do.
ModelWithGaussian.register("default", constructor="from_lazy_objects")(
    ModelWithGaussian
)
ModelWithGaussian.default_implementation = "default"


param_str = """{"gaussian": {"mean": 0.0, "variance": 1.0}}"""
params = Params(json.loads(param_str))

model = ModelWithGaussian.from_params(params=params)
print("Mean:", model.gaussian.mean)
print("Variance:", model.gaussian.variance)
