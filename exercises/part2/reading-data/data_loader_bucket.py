# Create a toy dataset from texts
texts = [['a', 'b', 'c', 'd'], ['e'], ['f', 'g', 'h'], ['i', 'j']]
texts = [[Token(t) for t in tokens] for tokens in texts]
token_indexers = {'tokens': SingleIdTokenIndexer()}
instances = [Instance({'tokens': TextField(tokens, token_indexers)}) for tokens in texts]
dataset = AllennlpDataset(instances)
vocab = Vocabulary.from_instances(dataset)
dataset.index_with(vocab)

print('Using a BasicBatchSampler:')
sampler = SequentialSampler(data_source=dataset)
batch_sampler = BasicBatchSampler(sampler, batch_size=2, drop_last=False)
data_loader = DataLoader(dataset, batch_sampler=batch_sampler)
for batch in data_loader:
    print(batch)

print('Using a BucketBatchSampler:')
# by default, a BucketBatchSampler tries to auto-detect the right sorting keys,
# but we are specifying them explicitly here
batch_sampler = BucketBatchSampler(dataset, batch_size=2, sorting_keys=['tokens'])
data_loader = DataLoader(dataset, batch_sampler=batch_sampler)
for batch in data_loader:
    print(batch)
