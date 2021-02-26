import sys
import os

sys.path.append(os.path.abspath(os.path.join("..", "nla_semparse")))

from nla_semparse.nla_metric import NlaMetric


def test_metric_basic():
    metric = NlaMetric()
    metric(["2"], ["2"])
    assert metric.get_metric() == {
        "well_formedness": 1.0,
        "denotation_accuracy": 1.0,
        "sequence_accuracy": 1.0,
    }
    metric.reset()


def test_metric_one_operation():
    metric = NlaMetric()
    metric(["(add 2 3)"], ["(add 2 3)"])
    assert metric.get_metric() == {
        "well_formedness": 1.0,
        "denotation_accuracy": 1.0,
        "sequence_accuracy": 1.0,
    }
    metric.reset()
    metric(["(add 2 3)"], ["5"])
    assert metric.get_metric() == {
        "well_formedness": 1.0,
        "denotation_accuracy": 1.0,
        "sequence_accuracy": 0.0,
    }
    metric.reset()
    metric(["(add 2 3)"], ["(add 1 4)"])
    assert metric.get_metric() == {
        "well_formedness": 1.0,
        "denotation_accuracy": 1.0,
        "sequence_accuracy": 0.0,
    }
    metric.reset()
    metric(["(add 2 3)"], ["(subtract 1 4)"])
    assert metric.get_metric() == {
        "well_formedness": 1.0,
        "denotation_accuracy": 0.0,
        "sequence_accuracy": 0.0,
    }
    metric.reset()


def test_metric_ill_formed_sequences():
    metric = NlaMetric()
    metric(["(add 2)"], ["(add 2 3)"])
    assert metric.get_metric() == {
        "well_formedness": 0.0,
        "denotation_accuracy": 0.0,
        "sequence_accuracy": 0.0,
    }
    metric.reset()
    metric(["(add 2))"], ["(add 2 3)"])
    assert metric.get_metric() == {
        "well_formedness": 0.0,
        "denotation_accuracy": 0.0,
        "sequence_accuracy": 0.0,
    }
    metric.reset()
    metric(["()"], ["(add 2 3)"])
    assert metric.get_metric() == {
        "well_formedness": 0.0,
        "denotation_accuracy": 0.0,
        "sequence_accuracy": 0.0,
    }
    metric.reset()


def test_metric_real_cases():
    predictions1 = [
        "(subtract (multiply (((((((((())))))",
        "(subtract (add ((multiply (((()))))))))",
    ]
    predictions2 = ["9", "9"]
    predictions3 = ["(subtract (multiply (((((((((())))))", "9"]
    targets = [
        "(add (add (multiply 5 2) (divide 2 7)) (add (add 7 7) (multiply 3 (multiply 6 6))))",
        "(subtract (add 8 7) (subtract (add (add 6 (divide 7 7)) 7) (multiply (divide 5 4) 8)))",
    ]
    metric = NlaMetric()
    metric(predictions1, targets)
    assert metric.get_metric() == {
        "well_formedness": 0.0,
        "denotation_accuracy": 0.0,
        "sequence_accuracy": 0.0,
    }
    metric.reset()
    metric(predictions2, targets)
    assert metric.get_metric() == {
        "well_formedness": 1.0,
        "denotation_accuracy": 0.5,
        "sequence_accuracy": 0.0,
    }
    metric.reset()
    metric(predictions3, targets)
    assert metric.get_metric() == {
        "well_formedness": 0.5,
        "denotation_accuracy": 0.5,
        "sequence_accuracy": 0.0,
    }
    metric.reset()
    metric(targets, targets)
    assert metric.get_metric() == {
        "well_formedness": 1.0,
        "denotation_accuracy": 1.0,
        "sequence_accuracy": 1.0,
    }
    metric.reset()
