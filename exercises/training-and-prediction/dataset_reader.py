CONFIG = """
{
    "dataset_reader" : {
        "type": "classification-tsv",
        "token_indexers": {
            "tokens": {
                "type": "single_id"
            }
        },
        "max_tokens": 64
    },
    "train_data_path": "quick_start/data/movie_review/train.tsv"
}
"""

_ = run_config(CONFIG)
