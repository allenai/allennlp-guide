# Inputs
text: TextField
title: TextField
stars: LabelField

# Outputs
aspect: List[LabelField]
sentiment: List[LabelField]  # or a SequenceLabelField that depends on `aspect`
