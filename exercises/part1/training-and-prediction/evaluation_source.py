# We've copied the training loop from an earlier example, with updated model
# code, above in the Setup section. We run the training loop to get a trained
# model.
model, dataset_reader = run_training_loop()

# Now we can evaluate the model on a new dataset.
test_data = list(dataset_reader.read("quick_start/data/movie_review/test.tsv"))
data_loader = SimpleDataLoader(test_data, batch_size=8)
data_loader.index_with(model.vocab)

results = evaluate(model, data_loader)
print(results)
