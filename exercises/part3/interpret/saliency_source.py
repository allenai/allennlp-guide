inputs = {"sentence": "a very well-made, funny and entertaining picture."}
archive = (
    "https://storage.googleapis.com/allennlp-public-models/"
    "basic_stanford_sentiment_treebank-2020.06.09.tar.gz"
)
predictor = Predictor.from_path(archive)
interpreter = SimpleGradient(predictor)
interpretation = interpreter.saliency_interpret_from_json(inputs)

print(interpretation)
