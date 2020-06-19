def evaluate(prediction: str, target: str):
    metric = NlaMetric()
    prediction_list = prediction.split(" ")
    target_list = target.split(" ")
    metric([prediction_list], [target_list])
    return  metric.get_metric(reset=True)


target = '( - ( * 7 3 ) 2 )'

predictions = [
    '( - ( * 7 3 ) 2 )',
    '( - ( * 6 4 ) 5 )',
    '- ( ) + /',
]

for prediction in predictions:
    metrics = evaluate(prediction, target)
    print(f"Prediction: {prediction}")
    print(f"Target: {target}")
    print(f"Well formedness: {metrics['well_formedness']}")
    print(f"Accuracy: {metrics['sequence_accuracy']}\n")
