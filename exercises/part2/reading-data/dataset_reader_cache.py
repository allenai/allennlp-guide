import os

from quick_start.my_text_classifier.dataset_readers import ClassificationTsvReader

# Instantiate a dataset reader with a cache directory
reader = ClassificationTsvReader(cache_directory='dataset_cache')
dataset = reader.read('quick_start/data/movie_review/train.tsv')

# Check if the dataset is cached
print(os.listdir('dataset_cache'))

# The next time you invoke read(), instances are read from the cache
dataset = reader.read('quick_start/data/movie_review/train.tsv')
