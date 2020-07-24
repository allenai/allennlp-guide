CUDA_DEVICE = 0  # Set device ID if you can use GPU.


def prepare_data():
    reader = TextClassificationJsonReader(
        token_indexers={"tokens": SingleIdTokenIndexer()},
        tokenizer=WhitespaceTokenizer(),
    )
    train_dataset = reader.read("https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/train.jsonl")  # NOQA
    valid_dataset = reader.read("https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/dev.jsonl")  # NOQA
    vocab = Vocabulary.from_instances(train_dataset)
    train_dataset.index_with(vocab)
    valid_dataset.index_with(vocab)
    return train_dataset, valid_dataset, vocab


def build_model(
        vocab: Vocabulary,
        embedding_dim: int,
        max_filter_size: int,
        num_filters: int,
        output_dim: int,
        dropout: float,
):
    model = BasicClassifier(
        text_field_embedder=BasicTextFieldEmbedder(
            {
                "tokens": Embedding(
                    embedding_dim=embedding_dim,
                    trainable=True,
                    vocab=vocab
                )
            }
        ),
        seq2vec_encoder=CnnEncoder(
            ngram_filter_sizes=range(2, max_filter_size),
            num_filters=num_filters,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
        ),
        dropout=dropout,
        vocab=vocab,
    )
    return model


def objective(trial: Trial):
    embedding_dim = trial.suggest_int("embedding_dim", 128, 256)
    max_filter_size = trial.suggest_int("max_filter_size", 3, 6)
    num_filters = trial.suggest_int("num_filters", 128, 256)
    output_dim = trial.suggest_int("output_dim", 128, 512)
    dropout = trial.suggest_float("dropout", 0, 1.0)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)

    train_dataset, valid_dataset, vocab = prepare_data()
    model = build_model(vocab, embedding_dim, max_filter_size, num_filters, output_dim, dropout)

    if CUDA_DEVICE > -1:
        model.to(torch.device("cuda:{}".format(CUDA_DEVICE)))

    optimizer = SGD(model.parameters(), lr=lr)
    data_loader = DataLoader(train_dataset, batch_size=10, collate_fn=allennlp_collate)
    validation_data_loader = DataLoader(valid_dataset, batch_size=64, collate_fn=allennlp_collate)
    trainer = GradientDescentTrainer(
        model=model,
        optimizer=optimizer,
        data_loader=data_loader,
        validation_data_loader=validation_data_loader,
        validation_metric="+accuracy",
        patience=None,  # `patience=None` since it could conflict with AllenNLPPruningCallback
        num_epochs=50,
        cuda_device=CUDA_DEVICE,
        serialization_dir="result",
        epoch_callbacks=[AllenNLPPruningCallback(trial, "validation_accuracy")],
    )
    return trainer.train()["best_validation_accuracy"]


def main():
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(),
    )
    study.optimize(objective, n_trials=50)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    random.seed(41)
    torch.manual_seed(41)
    numpy.random.seed(41)

    main()
