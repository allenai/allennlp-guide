# Create a toy model that just returns a random distribution over labels
class ToyModel(Model):
    def __init__(self, vocab: Vocabulary):
        super().__init__(vocab)

    def forward(self,
                tokens: TextFieldTensors,
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # Simply generate random logits and compute a probability distribution
        batch_size = label.size(0)
        logits = torch.normal(mean=0., std=1., size=(batch_size, 2))
        probs = torch.softmax(logits, dim=1)

        return {'logits': logits, 'probs': probs}

    def make_output_human_readable(
        self,
        output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # Take the logits from the forward pass, and compute the label
        # IDs for maximum values
        logits = output_dict['logits'].cpu().data.numpy()
        predicted_id = numpy.argmax(logits, axis=-1)
        # Convert these IDs back to label strings using vocab
        output_dict['label'] = [
            self.vocab.get_token_from_index(x, namespace='labels')
            for x in predicted_id
        ]
        return output_dict


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

# Run forward pass on an instance. This will invoke forward() then decode()
print(model.forward_on_instance(instance_pos))
# Notice that the return value is one dictionary per instance,
# even though everything in forward() and decode() is batched
print(model.forward_on_instances([instance_pos, instance_neg]))
