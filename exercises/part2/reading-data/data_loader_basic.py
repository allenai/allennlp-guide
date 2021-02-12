reader = MyDatasetReader()
vocab = Vocabulary.from_instances(reader.read("path_to_data"))

print("Default:")
data_loader = MultiProcessDataLoader(reader, "path_to_data", batch_size=4)
data_loader.index_with(vocab)
for batch in data_loader:
    print(batch)

print("Shuffle, and drop last batch if incomplete:")
data_loader = MultiProcessDataLoader(
    reader, "path_to_data", batch_size=4, shuffle=True, drop_last=True
)
data_loader.index_with(vocab)
for batch in data_loader:
    print(batch)
