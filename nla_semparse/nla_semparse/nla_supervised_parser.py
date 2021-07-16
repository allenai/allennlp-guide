from typing import Dict, List

from overrides import overrides
import torch

from allennlp.common.util import START_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder, Embedding
from allennlp.models.model import Model
from allennlp.nn import util, Activation
from allennlp_semparse.fields.production_rule_field import ProductionRule
from allennlp_semparse.state_machines import BeamSearch
from allennlp_semparse.state_machines.trainers import MaximumMarginalLikelihood
from allennlp_semparse.state_machines.transition_functions import BasicTransitionFunction
from allennlp_semparse.state_machines.states import GrammarBasedState, GrammarStatelet, RnnStatelet

from nla_semparse.nla_metric import NlaMetric
from nla_semparse.nla_language import NlaLanguage


@Model.register("nla-supervised")
class NlaSupervisedParser(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 input_embedder: TextFieldEmbedder,
                 action_embedding_dim: int,
                 encoder: Seq2SeqEncoder,
                 attention: Attention,
                 decoder_beam_search: BeamSearch,
                 max_decoding_steps: int,
                 dropout: float=0.0) -> None:
        super().__init__(vocab)
        self._input_embedder = input_embedder
        self._metric = NlaMetric()
        self._encoder = encoder
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        self._nla_world = NlaLanguage()
        self._action_embedder = Embedding(num_embeddings=len(self._nla_world.all_possible_productions()),
                                          embedding_dim=action_embedding_dim)

        # This is what we pass as input in the first step of decoding, when we don't have a
        # previous action.
        self._first_action_embedding = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))
        torch.nn.init.normal_(self._first_action_embedding)
        self._decoder_trainer = MaximumMarginalLikelihood()
        self._decoder_step = BasicTransitionFunction(encoder_output_dim=self._encoder.get_output_dim(),
                                                     action_embedding_dim=action_embedding_dim,
                                                     input_attention=attention,
                                                     activation=Activation.by_name("tanh")(),
                                                     add_action_bias=False,
                                                     dropout=dropout)
        self._decoder_beam_search = decoder_beam_search
        self._max_decoding_steps = max_decoding_steps
        self._action_padding_index = -1

    def _get_initial_grammar_statelet(self):
        nonterminal_keyed_actions = self._nla_world.get_nonterminal_productions()
        action_mapping = {action: i for i, action in enumerate(self._nla_world.all_possible_productions())}

        translated_valid_actions: Dict[str, Dict[str,Tuple[torch.Tensor, torch.Tensor, List[int]]]] = {}

        for key, action_strings in nonterminal_keyed_actions.items():
            translated_valid_actions[key] = {}
            # `key` here is a non-terminal from the grammar, and `action_strings` are all the valid
            # productions of that non-terminal.
            action_indices = [action_mapping[action_string] for action_string in action_strings]

            action_input_embeddings = self._action_embedder(torch.tensor(action_indices))
            translated_valid_actions[key]["global"] = (action_input_embeddings,
                                                       action_input_embeddings,
                                                       list(action_indices))
        initial_grammar_statelet = GrammarStatelet([START_SYMBOL],
                                                   translated_valid_actions,
                                                   self._nla_world.is_nonterminal)
        return initial_grammar_statelet

    def _get_initial_rnn_state(self, input_: Dict[str, torch.LongTensor]):
        embedded_input = self._input_embedder(input_)
        # (batch_size, input_length)
        input_mask = util.get_text_field_mask(input_)

        batch_size = embedded_input.size(0)

        # (batch_size, input_length, encoder_output_dim)
        encoder_outputs = self._dropout(self._encoder(embedded_input, input_mask))

        final_encoder_output = util.get_final_encoder_states(encoder_outputs,
                                                             input_mask,
                                                             self._encoder.is_bidirectional())
        memory_cell = encoder_outputs.new_zeros(batch_size, self._encoder.get_output_dim())
        attended_input, _ = self._decoder_step.attend_on_question(final_encoder_output,
                                                                  encoder_outputs,
                                                                  input_mask)
        encoder_outputs_list = [encoder_outputs[i] for i in range(batch_size)]
        input_mask_list = [input_mask[i] for i in range(batch_size)]
        initial_rnn_state = []
        for i in range(batch_size):
            initial_rnn_state.append(RnnStatelet(final_encoder_output[i],
                                                 memory_cell[i],
                                                 self._first_action_embedding,
                                                 attended_input[i],
                                                 encoder_outputs_list,
                                                 input_mask_list))
        return initial_rnn_state

    def _get_action_strings(self,
                            action_sequence_indices: List[List[int]]) -> List[List[str]]:
        all_productions = self._nla_world.all_possible_productions()
        action_sequences = []
        for instance_indices in action_sequence_indices:
            action_sequences.append([])
            for index in instance_indices:
                if index != -1:
                    action_sequences[-1].append(all_productions[index])

        return action_sequences

    @overrides
    def get_metrics(self, reset: bool=False) -> Dict[str, float]:
        return self._metric.get_metric(reset)

    @overrides
    def forward(self,  # type: ignore
                input_expression: Dict[str, torch.LongTensor],
                target_action_sequences: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """
        Decoder logic for producing type constrained target sequences, trained to maximize likelihod
        of target action sequences.
        """
        batch_size = input_expression["tokens"]["tokens"].size(0)

        initial_rnn_state = self._get_initial_rnn_state(input_expression)
        token_ids = util.get_token_ids_from_text_field_tensors(input_expression)
        initial_score_list = [token_ids.new_zeros(1, dtype=torch.float) for i in range(batch_size)]
        initial_grammar_state = [self._get_initial_grammar_statelet() for i in range(batch_size)]

        production_rules = [[ProductionRule(action, is_global_rule=True)
                             for action in self._nla_world.all_possible_productions()]] * batch_size
        initial_state = GrammarBasedState(batch_indices=list(range(batch_size)),
                                          action_history=[[] for _ in range(batch_size)],
                                          score=initial_score_list,
                                          rnn_state=initial_rnn_state,
                                          grammar_state=initial_grammar_state,
                                          possible_actions=production_rules)

        if target_action_sequences is not None:
            # Remove the trailing dimension (from ListField[ListField[IndexField]]).
            target_action_sequences = target_action_sequences.squeeze(-1).unsqueeze(1)
            target_mask = target_action_sequences != self._action_padding_index
            outputs = self._decoder_trainer.decode(initial_state,
                                                   self._decoder_step,
                                                   (target_action_sequences,
                                                    target_mask))
        else:
            target_mask = None
            outputs: Dict[str, torch.Tensor] = {}

        if not self.training:
            best_final_states = self._decoder_beam_search.search(self._max_decoding_steps,
                                                                 initial_state,
                                                                 self._decoder_step,
                                                                 keep_final_unfinished_states=False)
            best_action_sequences: List[List[int]] = []
            for i in range(batch_size):
                # Decoding may not have terminated with any completed logical forms, if `num_steps`
                # isn't long enough (or if the model is not trained enough and gets into an
                # infinite action loop).
                if i in best_final_states:
                    best_action_indices = [best_final_states[i][0].action_history[0]]
                    best_action_sequences.append(best_action_indices[0])
                else:
                    best_action_sequences.append([])

            predicted_action_strings = self._get_action_strings(best_action_sequences)
            predicted_expressions = [self._nla_world.action_sequence_to_logical_form(sequence)
                                     for sequence in predicted_action_strings]

            if target_action_sequences is not None:
                target_index_lists = [[int(x) for x in instance_indices] for instance_indices in
                                      target_action_sequences.squeeze(1).detach().cpu().numpy()]
                target_action_strings = self._get_action_strings(target_index_lists)
                target_expressions = [self._nla_world.action_sequence_to_logical_form(sequence)
                                      for sequence in target_action_strings]
                self._metric(predicted_expressions, target_expressions)
            outputs["predicted_expressions"] = predicted_expressions
        return outputs
