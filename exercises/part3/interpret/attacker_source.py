inputs = {"sentence": "a very well-made, funny and entertaining picture."}
archive = "https://storage.googleapis.com/allennlp-public-models/basic_stanford_sentiment_treebank-2020.06.09.tar.gz"
predictor = Predictor.from_path(archive)
reducer = InputReduction(predictor)  # or Hotflip(predictor)
# if it is Hotflip, we need an extra step: reducer.initialize()
reduced = reducer.attack_from_json(inputs, "tokens", "grad_input_1")

print(reduced)
