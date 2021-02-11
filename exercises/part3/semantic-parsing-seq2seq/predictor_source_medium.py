inputs = ["one plus three minus four", "five minus six times seven over one"]


for nla_input in inputs:
    output = translate_nla(nla_input)
    print(f"Input: {nla_input}")
    print(f"Prediction: {output}\n")
