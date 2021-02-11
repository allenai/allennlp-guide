def evaluate(prediction: str, target: str) -> Dict[str, float]:
    metric = NlaMetric()
    metric([prediction], [target])
    return metric.get_metric(reset=True)


target = "(subtract (multiply 7 3) 2)"

predictions = [
    "(subtract (multiply 7 3) 2)",
    "(subtract (multiply 6 4) 5)",
    "subtract () add divide",
]

for prediction in predictions:
    metrics = evaluate(prediction, target)
    print(f"Prediction: {prediction}")
    print(f"Target: {target}")
    print(f"Well formedness: {metrics['well_formedness']}")
    print(f"Accuracy: {metrics['sequence_accuracy']}\n")
