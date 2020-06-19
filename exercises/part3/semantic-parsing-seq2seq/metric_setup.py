from typing import Dict, List, Optional
from overrides import overrides
from allennlp.training.metrics.metric import Metric


class NlaMetric(Metric):
    def __init__(self):
        self._operator_logic = {"+": lambda x, y: x + y,
                                "-": lambda x, y: x - y,
                                "*": lambda x, y: x * y,
                                "/": lambda x, y: x // y}
        self._num_well_formed = 0
        self._num_correct_denotation = 0
        self._num_same_sequence = 0
        self._num_all_sequences = 0

    def _get_bracketed_argument(self,
                                sequence: List[str],
                                starting_index: int) -> Optional[List[str]]:
        if len(sequence) <= starting_index:
            return None
        if sequence[starting_index] == ')' or \
           sequence[starting_index] in self._operator_logic:
            return None
        if sequence[starting_index] != '(':
            # This must be an integer.
            return [sequence[starting_index]]
        argument = []
        parentheses_stack = []
        for element in sequence[starting_index:]:
            argument.append(element)
            if element == '(':
                parentheses_stack.insert(len(parentheses_stack), element)
            elif element == ')':
                if not parentheses_stack:
                    return None
                parentheses_stack.pop()
                if not parentheses_stack:
                    return argument
        if parentheses_stack:
            return None

    def _evaluate(self, sequence: List[str]) -> Optional[int]:
        if len(sequence) == 1:
            try:
                return int(sequence[0])
            except ValueError:
                return None
        if sequence[0] != '(' or sequence[-1] != ')':
            return None
        if len(sequence) == 2:
            return None
        rest = sequence[1:-1]
        if rest[0] in self._operator_logic:
            first_argument = self._get_bracketed_argument(rest, 1)
            if first_argument is None:
                return None
            evaluated_first_argument = self._evaluate(first_argument)
            if evaluated_first_argument is None:
                return None
            second_argument = self._get_bracketed_argument(rest,
                                                           len(first_argument) + 1)
            if second_argument is None:
                return None
            evaluated_second_argument = self._evaluate(second_argument)
            if evaluated_second_argument is None:
                return None
            try:
                return self._operator_logic[rest[0]](evaluated_first_argument,
                                                     evaluated_second_argument)
            except ZeroDivisionError:
                return 0  # Not None because the sequence is well-formed.
        return None


    @overrides
    def __call__(self, predictions, targets) -> None:
        for predicted_sequence, target_sequence in zip(predictions, targets):
            evaluated_prediction = self._evaluate(predicted_sequence)
            evalauted_target = self._evaluate(target_sequence)
            if evaluated_prediction is not None:
                self._num_well_formed += 1
            if evaluated_prediction == evalauted_target:
                self._num_correct_denotation += 1
            if predicted_sequence == target_sequence:
                self._num_same_sequence += 1
            self._num_all_sequences += 1

    @overrides
    def get_metric(self, reset: bool=False) -> Dict[str, float]:
        if self._num_all_sequences == 0:
            metrics = {"well_formedness": 0.0,
                       "denotation_accuracy": 0.0,
                       "sequence_accuracy": 0.0}
        else:
            metrics = {
                "well_formedness": self._num_well_formed / self._num_all_sequences,
                "denotation_accuracy": self._num_correct_denotation / self._num_all_sequences,
                "sequence_accuracy": self._num_same_sequence / self._num_all_sequences
            }
        if reset:
            self.reset()
        return metrics

    @overrides
    def reset(self):
        self._num_well_formed = 0
        self._num_same_sequence = 0
        self._num_correct_denotation = 0
        self._num_all_sequences = 0
