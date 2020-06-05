class Seq2SeqDatasetReader(DatasetReader):
    def __init__(
        self,
        source_tokenizer: Tokenizer = None,
        target_tokenizer: Tokenizer = None,
        source_token_indexers: Dict[str, TokenIndexer] = None,
        target_token_indexers: Dict[str, TokenIndexer] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._source_tokenizer = source_tokenizer or WhitespaceTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = (source_token_indexers
                                       or {"tokens": SingleIdTokenIndexer()})
        self._target_token_indexers = (target_token_indexers
                                       or self._source_token_indexers)

    def _read(self, file_path: str):
        with open(cached_path(file_path), "r") as data_file:
            for line_num, row in enumerate(csv.reader(data_file, delimiter='\t')):
                source_sequence, target_sequence = row
                yield self.text_to_instance(source_sequence, target_sequence)

    def text_to_instance(
        self, source_string: str, target_string: str = None
    ) -> Instance:

        tokenized_source = self._source_tokenizer.tokenize(source_string)
        source_field = TextField(tokenized_source, self._source_token_indexers)
        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(
                tokenized_target,
                self._target_token_indexers
            )
            return Instance({
                "source_tokens": source_field,
                "target_tokens": target_field
            })
        else:
            return Instance({"source_tokens": source_field})

source_token_indexers = {
    "tokens": SingleIdTokenIndexer(namespace="source_tokens")
}
target_token_indexers = {
    "tokens": SingleIdTokenIndexer(namespace="target_tokens")
}
dataset_reader = Seq2SeqDatasetReader(
    source_token_indexers=source_token_indexers,
    target_token_indexers=target_token_indexers,
)
instances = dataset_reader.read("TODO")

for instance in instances[:10]:
    print(instance)
