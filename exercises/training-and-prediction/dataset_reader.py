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
    "train_data_path": "exercises/your-first-model/train.tsv"
}
"""

_ = run_config(CONFIG)
