from overrides import overrides
from typing import Dict

from allennlp.data import DatasetReader
from allennlp.data.fields import Field, TextField, ListField, IndexField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer

from nla_semparse.nla_language import NlaLanguage

@DatasetReader.register("nla")
class NlaDatasetReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 input_token_indexers: Dict[str, TokenIndexer] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._input_token_indexers = input_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._nla_language = NlaLanguage()

    @overrides
    def _read(self, file_path: str):
        with open(file_path, "r") as data_file:
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                line_parts = line.split("\t")
                if len(line_parts) != 2:
                    raise RuntimeError("Unexpected data format: {line}\nExpected tsv with two columns")
                input_expression, target = line_parts
                instance = self.text_to_instance(input_expression, target)
                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(self,
                         input_expression: str,
                         target_expression: str) -> Instance:
        tokenized_input = self._tokenizer.tokenize(input_expression)
        input_field = TextField(tokenized_input, self._input_token_indexers)
        all_productions = self._nla_language.all_possible_productions()
        all_productions_field = ListField([LabelField(label) for label in all_productions])
        production_indices = {production: i for i, production in enumerate(all_productions)}

        # The language expects expressions to not have padding spaces withing parenteses.
        target_expression = target_expression.replace("( ", "(").replace(" )", ")")
        target_action_sequence = self._nla_language.logical_form_to_action_sequence(target_expression)
        target_action_index_fields = [IndexField(production_indices[action], all_productions_field)
                                      for action in target_action_sequence]
        target_action_field = ListField(target_action_index_fields)
        return Instance({"input_expression": input_field,
                         "target_action_sequences": target_action_field})
