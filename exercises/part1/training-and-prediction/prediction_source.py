class SentenceClassifierPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)


# We've copied the training loop from an earlier example, with updated model
# code, above in the Setup section. We run the training loop to get a trained
# model.
model, dataset_reader = run_training_loop()
vocab = model.vocab
predictor = SentenceClassifierPredictor(model, dataset_reader)

output = predictor.predict('A good movie!')
print([(vocab.get_token_from_index(label_id, 'labels'), prob)
       for label_id, prob in enumerate(output['probs'])])
output = predictor.predict('This was a monstrous waste of time.')
print([(vocab.get_token_from_index(label_id, 'labels'), prob)
       for label_id, prob in enumerate(output['probs'])])
