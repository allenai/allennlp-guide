import json
from allennlp.common import FromParams, Params


class BaseGaussian(FromParams):
    def __init__(self, mean: float, variance: float):
        self.mean = mean
        self.variance = variance


class MyGaussian(BaseGaussian):
    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name


param_str = """{"mean": 0.0, "variance": 1.0, "name": "My Gaussian"}"""
params = Params(json.loads(param_str))
gaussian = MyGaussian.from_params(params)
print(f"Mean: {gaussian.mean}")
print(f"Variance: {gaussian.variance}")
print(f"Name: {gaussian.name}")
