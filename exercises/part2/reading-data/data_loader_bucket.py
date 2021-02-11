reader = MyDatasetReader()
vocab = Vocabulary.from_instances(reader.read("path_to_data"))

print("Using the BucketBatchSampler:")
# The sorting_keys argument is unnecessary here, because the sampler will
# automatically detect that 'tokens' is the right sorting key, but we are
# including it in our example for completeness. You can remove it and see
# that the output is the same.
data_loader = MultiProcessDataLoader(
    reader,
    "path_to_data",
    batch_sampler=BucketBatchSampler(batch_size=4, sorting_keys=["tokens"]),
)
data_loader.index_with(vocab)
for batch in data_loader:
    print(batch)
