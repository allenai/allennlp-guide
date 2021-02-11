import torch
from allennlp.nn.initializers import ConstantInitializer
from allennlp.nn.regularizers import L1Regularizer, L2Regularizer, RegularizerApplicator


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = torch.nn.Linear(3, 2)
        self.conv = torch.nn.Conv1d(2, 2, 2)

    def forward(self, inputs):
        pass


print("Using individual regularizers:")
model = Net()
init_const = ConstantInitializer(val=10.0)
init_const(model.linear1.weight)
init_const(model.linear2.weight)

l1_regularizer = L1Regularizer(alpha=0.01)
print(l1_regularizer(model.linear1.weight))  # 0.01 * 10 * 6 = 0.6

l2_regularizer = L2Regularizer(alpha=0.01)
print(l2_regularizer(model.linear2.weight))  # 0.01 * (10)^2 * 6

print("Using an applicator:")
applicator = RegularizerApplicator(
    regexes=[
        ("linear1.weight", L1Regularizer(alpha=0.01)),
        ("linear2.weight", L2Regularizer()),
    ]
)
print(applicator(model))  # 0.6 + 6
