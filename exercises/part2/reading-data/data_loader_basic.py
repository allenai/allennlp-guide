# Create a toy dataset from labels
instances = [Instance({'label': LabelField(str(label))})
             for label in 'abcdefghij']
dataset = AllennlpDataset(instances)
vocab = Vocabulary.from_instances(dataset)
dataset.index_with(vocab)

# Use the default batching mechanism
print("Default:")
data_loader = DataLoader(dataset, batch_size=3)
for batch in data_loader:
    print(batch)

# Use Samplers to customize the sequencing / batching behavior
sampler = SequentialSampler(data_source=dataset)
batch_sampler = BasicBatchSampler(sampler, batch_size=3, drop_last=True)

print("\nDropping last:")
data_loader = DataLoader(dataset, batch_sampler=batch_sampler)
for batch in data_loader:
    print(batch)

# Example: using a RandomSampler instead of a SequentialSampler
sampler = RandomSampler(data_source=dataset)
batch_sampler = BasicBatchSampler(sampler, batch_size=3, drop_last=False)

print("\nWith RandomSampler:")
data_loader = DataLoader(dataset, batch_sampler=batch_sampler)
for batch in data_loader:
    print(batch)
