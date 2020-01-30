from itertools import islice
from quick_start.my_text_classifier.dataset_readers import ClassificationTsvReader

reader = ClassificationTsvReader(lazy=True)
dataset = reader.read('quick_start/data/movie_review/train.tsv')

# Returned dataset is a wrapped by a `LazyInstances` object
print('type of dataset: ', type(dataset))

# Preview the first 5 instances. This is when the _read() method is called for the first time
print('first 5 instances: ', list(islice(dataset, 5)))
