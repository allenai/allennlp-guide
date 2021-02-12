# Create Fields
tokens = [Token("The"), Token("best"), Token("movie"), Token("ever"), Token("!")]
token_indexers: Dict[str, TokenIndexer] = {"tokens": SingleIdTokenIndexer()}
text_field = TextField(tokens, token_indexers=token_indexers)

label_field = LabelField("pos")

sequence_label_field = SequenceLabelField(
    ["DET", "ADJ", "NOUN", "ADV", "PUNKT"], text_field
)

# Create an Instance
fields: Dict[str, Field] = {
    "tokens": text_field,
    "label": label_field,
}
instance = Instance(fields)

# You can add fields later
instance.add_field("label_seq", sequence_label_field)

# You can simply use print() to see the instance's content
print(instance)

# Create a Vocabulary
counter = defaultdict(Counter)
instance.count_vocab_items(counter)
vocab = Vocabulary(counter)

# Convert all strings in all of the fields into integer IDs by calling index_fields()
instance.index_fields(vocab)

# Instances know how to turn themselves into a dict of tensors.  When we call this
# method in our data code, we additionally give a `padding_lengths` argument.
# We will pass this dictionary to the model as **tensors, so be sure the keys
# match what the model expects.
tensors = instance.as_tensor_dict()
print(tensors)
