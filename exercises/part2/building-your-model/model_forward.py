# Create a toy model that just prints tensors passed to forward
class ToyModel(Model):
    def __init__(self, vocab: Vocabulary):
        super().__init__(vocab)

    # Note that the signature of forward() needs to match that of field names
    def forward(self,
                tokens: TextFieldTensors,
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        print('tokens:', tokens)
        print('label:', label)

        return {}


# Create fields and instances
token_indexers = {'tokens': SingleIdTokenIndexer(namespace='tokens')}
text_field_pos = TextField(
    [Token('The'), Token('best'), Token('movie'), Token('ever'), Token('!')],
    token_indexers=token_indexers)
text_field_neg = TextField(
    [Token('Such'), Token('an'), Token('awful'), Token('movie'), Token('.')],
    token_indexers=token_indexers)

label_field_pos = LabelField('pos', label_namespace='labels')
label_field_neg = LabelField('neg', label_namespace='labels')

instance_pos = Instance({'tokens': text_field_pos, 'label': label_field_pos})
instance_neg = Instance({'tokens': text_field_neg, 'label': label_field_neg})
instances = [instance_pos, instance_neg]


# Create a Vocabulary
vocab = Vocabulary.from_instances(instances)

dataset = AllennlpDataset(instances, vocab)

# Create an iterator that creates batches of size 2
data_loader = DataLoader(dataset, batch_size=2)

model = ToyModel(vocab)

# Iterate over batches and pass them to forward()
for batch in data_loader:
    model(**batch)
