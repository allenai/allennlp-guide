inputs = ['seven times two plus three minus six over two plus four',
          'eight over nine times six minus three plus seven over five minus one']


for nla_input in inputs:
    output = translate_nla(nla_input)
    print(f"Input: {nla_input}")
    print(f"Prediction: {output}\n")
