def run_training_loop():
    dataset_reader = build_dataset_reader()

    train_data, dev_data = read_data(dataset_reader)

    vocab = build_vocab(train_data + dev_data)
    model = build_model(vocab)

    train_loader, dev_loader = build_data_loaders(train_data, dev_data)
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)

    # You obviously won't want to create a temporary file for your training
    # results, but for execution in binder for this guide, we need to do this.
    with tempfile.TemporaryDirectory() as serialization_dir:
        trainer = build_trainer(model, serialization_dir, train_loader, dev_loader)
        print("Starting training")
        trainer.train()
        print("Finished training")

    return model, dataset_reader


def build_data_loaders(
    train_data: List[Instance],
    dev_data: List[Instance],
) -> Tuple[DataLoader, DataLoader]:
    train_loader = SimpleDataLoader(train_data, 8, shuffle=True)
    dev_loader = SimpleDataLoader(dev_data, 8, shuffle=False)
    return train_loader, dev_loader


def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader,
) -> Trainer:
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters)  # type: ignore
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=5,
        optimizer=optimizer,
    )
    return trainer


run_training_loop()
