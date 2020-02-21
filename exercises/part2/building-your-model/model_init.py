import torch
from allennlp.common import Params
from allennlp.nn import InitializerApplicator, Initializer
import json


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = torch.nn.Linear(3, 2)
        self.conv = torch.nn.Conv1d(2, 2, 2)

    def forward(self, inputs):
        pass


model = Net()
print('Initial parameters:')
for name, param in model.named_parameters():
    print(name, param)

init_uniform = Initializer.from_params(Params({'type': 'xavier_uniform'}))
init_uniform(model.linear1.weight)
init_uniform(model.linear2.weight)

init_const = Initializer.from_params(Params({'type': 'constant', 'val': 10}))
init_const(model.linear1.bias)
init_const(model.linear2.bias)

init_normal = Initializer.from_params(Params({'type': 'normal', 'mean': 0., 'std': 10.}))
init_normal(model.conv.weight)
init_normal(model.conv.bias)

print('After applying initializers individually:')
for name, param in model.named_parameters():
    print(name, param)


config = """
{"initializer":
    [
        ["linear.*weight", {"type": "xavier_uniform"}],
        ["linear.*bias", {"type": "constant", "val": 10}],
        ["conv.*", {"type": "normal", "mean": 0.0, "std": 10.0}]
    ]
}
"""
model = Net()
params = Params(json.loads(config))
applicator = InitializerApplicator.from_params(params['initializer'])
applicator(model)

print('After applying an applicator:')
for name, param in model.named_parameters():
    print(name, param)
