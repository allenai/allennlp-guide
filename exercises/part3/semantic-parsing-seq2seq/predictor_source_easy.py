inputs = ['one plus three',
          'five minus six',
          'seven times two',
          'four over nine']


for nla_input in inputs:
    output = translate_nla(nla_input)
    print(f"Input: {nla_input}")
    print(f"Prediction: {output}\n")
