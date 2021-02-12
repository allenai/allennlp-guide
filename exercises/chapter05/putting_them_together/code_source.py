from allennlp.data import DatasetReader, Instance, Vocabulary
from allennlp.data.fields import LabelField, TextField
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.models import Model
from allennlp.modules import Embedding, Seq2VecEncoder, TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder

# Data will be formatted as:
# [title][tab][text][tab][stars][tab][aspect][tab][sentiment]


@DatasetReader.register("classification-tsv")
class ClassificationTsvReader(DatasetReader):
    def __init__(self):
        self.tokenizer = WordTokenizer()
        self.token_indexers = {"tokens": SingleIdTokenIndexer()}

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, "r") as lines:
            for line in lines:
                text, sentiment = line.strip().split("\t")
                text_field = TextField(
                    self.tokenizer.tokenize(text), self.token_indexers
                )
                sentiment_field = LabelField(sentiment)
                fields = {"text": text_field, "sentiment": sentiment_field}
                yield Instance(fields)


reader = ClassificationTsvReader()
instances = reader.read("exercises/chapter05/input_output_reader/example_data.tsv")
print(instances)

vocab = Vocabulary.from_instances(instances)
print(vocab)


@Model.register("simple_classifier")
class SimpleClassifier(Model):
    def __init__(
        self, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder
    ):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)

    def forward(
        self, text: Dict[str, torch.Tensor], label: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        loss = torch.nn.functional.cross_entropy(logits, label)
        return {"loss": loss, "probs": probs}


iterator = BasicIterator(batch_size=2)
iterator.index_with(vocab)

embedding = Embedding(num_embeddings=vocab.get_vocab_size("tokens"), embedding_dim=10)

model = SimpleClassifier(
    vocab, BasicTextFieldEmbedder({"tokens": embedding}), BagOfEmbeddingsEncoder()
)

for batch in iterator(instances, num_epochs=1):
    outputs = model(batch)
    print(f"Model outputs: {outputs}")
