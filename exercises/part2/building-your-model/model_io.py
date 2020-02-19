CONFIG = """
{
    "dataset_reader" : {
        "type": "classification-tsv",
        "token_indexers": {
            "tokens": {
                "type": "single_id"
            }
        }
    },
    "train_data_path": "quick_start/data/movie_review/train.tsv",
    "validation_data_path": "quick_start/data/movie_review/dev.tsv",
    "model": {
        "type": "simple_classifier",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 10
                }
            }
        },
        "encoder": {
            "type": "bag_of_embeddings",
            "embedding_dim": 10
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size": 8
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 5
    }
}
"""


def make_predictions(model: Model, dataset_reader: DatasetReader) \
        -> List[Dict[str, float]]:
    """Make predictions using the given model and dataset reader."""
    predictions = []
    predictor = SentenceClassifierPredictor(model, dataset_reader)
    output = predictor.predict('A good movie!')
    predictions.append({vocab.get_token_from_index(label_id, 'labels'): prob
                        for label_id, prob in enumerate(output['probs'])})
    output = predictor.predict('This was a monstrous waste of time.')
    predictions.append({vocab.get_token_from_index(label_id, 'labels'): prob
                        for label_id, prob in enumerate(output['probs'])})
    return predictions


components = run_config(CONFIG)
params, dataset_reader, vocab, model = components['params'], components['dataset_reader'], components['vocab'], components['model']


original_preds = make_predictions(model, dataset_reader)

# Save the model
serialization_dir = 'model'
config_file = os.path.join(serialization_dir, 'config.json')
vocabulary_dir = os.path.join(serialization_dir, 'vocabulary')
weights_file = os.path.join(serialization_dir, 'weights.th')

os.makedirs(serialization_dir, exist_ok=True)
params.to_file(config_file)
vocab.save_to_files(vocabulary_dir)
torch.save(model.state_dict(), weights_file)

# Load the model
loaded_params = Params.from_file(config_file)
loaded_model = Model.load(loaded_params, serialization_dir, weights_file)
loaded_vocab = loaded_model.vocab   # Vocabulary is loaded in Model.load()

# Make sure the predictions are the same
loaded_preds = make_predictions(loaded_model, dataset_reader)
assert original_preds == loaded_preds
print('predictions matched')

# Create an archive file
archive_model(serialization_dir, weights='weights.th')

# Unarchive from the file
archive = load_archive(os.path.join(serialization_dir, 'model.tar.gz'))

# Make sure the predictions are the same
archived_preds = make_predictions(archive.model, dataset_reader)
assert original_preds == archived_preds
print('predictions matched')
