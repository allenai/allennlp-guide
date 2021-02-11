config = {
    "dataset_reader": {
        "type": "classification-tsv",
        "token_indexers": {"tokens": {"type": "single_id"}},
    },
    "train_data_path": "quick_start/data/movie_review/train.tsv",
    "validation_data_path": "quick_start/data/movie_review/dev.tsv",
    "model": {
        "type": "simple_classifier",
        "embedder": {
            "token_embedders": {"tokens": {"type": "embedding", "embedding_dim": 10}}
        },
        "encoder": {"type": "bag_of_embeddings", "embedding_dim": 10},
    },
    "data_loader": {"batch_size": 8, "shuffle": True},
    "trainer": {"optimizer": "adam", "num_epochs": 5},
}


with tempfile.TemporaryDirectory() as serialization_dir:
    config_filename = serialization_dir + "/training_config.json"
    with open(config_filename, "w") as config_file:
        json.dump(config, config_file)
    from allennlp.commands.train import train_model_from_file

    # Instead of this python code, you would typically just call
    # allennlp train [config_file] -s [serialization_dir]
    train_model_from_file(
        config_filename, serialization_dir, file_friendly_logging=True, force=True
    )
