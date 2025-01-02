# classEval

ds = open("data/codesearchnet/python_valid_0.jsonl", 'r')

lines = ds.readlines()

print(len(lines))

for i, line in enumerate(lines):

    js = eval(line)
    #
    # print(eval(lines)['code_tokens'])
    # code = ' '.join(js['code_tokens']).replace('\n', ' ')
    # code = ' '.join(code.strip().split())
    # print(code)
    nl = ' '.join(js['docstring_tokens']).replace('\n', '')
    nl = ' '.join(nl.strip().split())
    print(nl)