from allennlp.data import DataLoader
from allennlp.data import Instance
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset
from allennlp.data.fields import LabelField
from allennlp.data.samplers import SequentialSampler, RandomSampler, BasicBatchSampler
from allennlp.data.vocabulary import Vocabulary

# Create a toy dataset
instances = [Instance({'label': LabelField(str(label))}) for label in range(10)]
dataset = AllennlpDataset(instances)
vocab = Vocabulary.from_instances(dataset)
dataset.index_with(vocab)

# Use the default batching mechanism
data_loader = DataLoader(dataset, batch_size=3)
for batch in data_loader:
    print(batch)

# Use Samplers to customize the sequencing / batching behavior
sampler = SequentialSampler(data_source=dataset)
batch_sampler = BasicBatchSampler(sampler, batch_size=3, drop_last=True)

data_loader = DataLoader(dataset, batch_sampler=batch_sampler)
for batch in data_loader:
    print(batch)

# Example: using a RandomSampler instead of a SequentialSampler
sampler = RandomSampler(data_source=dataset)
batch_sampler = BasicBatchSampler(sampler, batch_size=3, drop_last=False)

data_loader = DataLoader(dataset, batch_sampler=batch_sampler)
for batch in data_loader:
    print(batch)
