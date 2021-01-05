from typing import Dict, List, Optional
from overrides import overrides
from allennlp.training.metrics.metric import Metric

from allennlp_semparse.domain_languages.domain_language import ExecutionError

from nla_semparse.nla_semparse.nla_language import NlaLanguage

@Metric.register('nla_metric')
class NlaMetric(Metric):
    """
    Metric for evaluating prefix arithmetic sequences against targets, useful for Natural Language Arithmetic
    parsing. This metric evaluates predicted sequences on three things: 1) whether the predicted metric is a
    well-formed prefix arithmetic expression, 2) whether the predicted sequence and the target seqquence evaluate
    to the same value, 3) whether the predicted sequence and the target sequence are identical.
    """
    def __init__(self):
        self._language = NlaLanguage()
        self._num_well_formed = 0
        self._num_correct_denotation = 0
        self._num_same_sequence = 0
        self._num_all_sequences = 0

    @overrides
    def __call__(self, predictions, targets) -> None:
        for prediction, target in zip(predictions, targets):
            if isinstance(prediction, list):
                prediction = " ".join(prediction).replace("( ", "(").replace(" )", ")")
                target = " ".join(target).replace("( ", "(").replace(" )", ")")
            if isinstance(prediction, str) and not prediction.startswith('('):
                prediction = f"({prediction})"
            if isinstance(target, str) and not target.startswith('('):
                target = f"({target})"

            evaluated_prediction = None
            evaluated_target = None
            try:
                evaluated_target = self._language.execute(target)
                evaluated_prediction = self._language.execute(prediction)
            except (TypeError, ExecutionError, IndexError):
                pass
            if isinstance(evaluated_prediction, int):
                self._num_well_formed += 1
            if evaluated_prediction == evaluated_target:
                self._num_correct_denotation += 1
            if prediction == target:
                self._num_same_sequence += 1
            self._num_all_sequences += 1

    @overrides
    def get_metric(self, reset: bool=False) -> Dict[str, float]:
        if self._num_all_sequences == 0:
            metrics = {"well_formedness": 0.0,
                       "denotation_accuracy": 0.0,
                       "sequence_accuracy": 0.0}
        else:
            metrics = {"well_formedness": self._num_well_formed / self._num_all_sequences,
                       "denotation_accuracy": self._num_correct_denotation / self._num_all_sequences,
                       "sequence_accuracy": self._num_same_sequence / self._num_all_sequences}
        if reset:
            self.reset()
        return metrics

    @overrides
    def reset(self):
        self._num_well_formed = 0
        self._num_same_sequence = 0
        self._num_correct_denotation = 0
        self._num_all_sequences = 0
