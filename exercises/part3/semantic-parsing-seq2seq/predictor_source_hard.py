inputs = [
    "eight over nine times six minus three plus seven over five minus one",
    "seven times eight plus five minus six plus one plus three plus two over seven",
]


for nla_input in inputs:
    output = translate_nla(nla_input)
    print(f"Input: {nla_input}")
    print(f"Prediction: {output}\n")
