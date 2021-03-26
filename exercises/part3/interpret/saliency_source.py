inputs = {"sentence": "a very well-made, funny and entertaining picture."}
archive = load_archive(
    "https://storage.googleapis.com/allennlp-public-models/basic_stanford_sentiment_treebank-2020.06.09.tar.gz"
)
predictor = Predictor.from_archive(archive)
interpreter = SimpleGradient(predictor)
interpretation = interpreter.saliency_interpret_from_json(inputs)

print(interpretation)
