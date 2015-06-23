
###############################################################################
# Model application
###############################################################################

import sys
import os
import network
from vec4net import make_vec
import numpy as np

# Replace this with your model's result
JSON = '../var/30hidden-30epochs-10batch-0.5eta-5.0lambda-7window-100shape-3run.net'
net = network.load(JSON)


def _get_iob(arr):
    d = {0: 'i', 1: 'o', 2: 'b'}
    n = np.argmax(arr)
    return d[n]

def _classify(token_list):
    "Classify a list of token"
    result = []
    sen_vec = make_vec(token_list)
    for x in sen_vec:
        result.append(_get_iob(net.feedforward(x)))
    return result

def _make_words(token_list, iob_list):
    "Make segmented words from token list and corresponding iob list"
    if not iob_list: return
    t = token_list[0:1]
    tokens = []
    for i in range(1, len(iob_list)):
        if iob_list[i] == 'i':
            t.append(token_list[i])
            continue
        if iob_list[i] == 'b':
            if not t:
                t = token_list[i:i+1]
                tokens.append(t)
                t = []
            else:
                tokens.append(t)
                t = token_list[i:i+1]
            continue
        if iob_list[i] == 'o':
            if t:
                tokens.append(t)
                t = []
            tokens.append(token_list[i:i+1])
    if t: tokens.append(t)
    return ['_'.join(tok) for tok in tokens]

def _test():
    tok = ['chủ', 'nhân', 'website', 'muốn', 'duy', 'trì', 'domain', '.com', 'phải', 'trả', '6,42', 'usd', '(', 'tăng', '7', '%', ')', ',', 'còn', '.net', 'sẽ', 'tốn', '3,85', 'usd', '(', 'tăng', '10', '%', ')', 'mỗi', 'năm', '.']
    iob = ['b', 'i', 'b', 'b', 'b', 'i', 'b', 'b', 'b', 'b', 'b', 'b', 'o', 'b', 'b', 'o', 'o', 'o', 'b', 'b', 'b', 'b', 'b', 'b', 'o', 'b', 'b', 'o', 'o', 'b', 'b', 'o']
    return _make_words(tok, iob)

if __name__ == '__main__':
    n_args = len(sys.argv)
    if n_args < 3:
        print("""
              VietSeg - Ver. 0.0.1
              ==================================
              Usage: python3 vietseg.py <input file> <output file>
               * The <input file> must be in UTF-8 encoding.
               * The segmented text will be written in the <output file>.
              ==================================
              http://github.com/manhtai/vietseg
              """)
        exit(1)
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    if not os.path.isfile(input_file):
        print('Input text file "' + input_file+ '" does not exist.')
        exit(1)
    with open(input_file, 'r') as fi, open(output_file, 'w') as fo:
        for line in fi:
            in_line = line.split()
            if not in_line:
                fo.write(line)
                continue
            token_list = line.lower().split()
            iob_list = _classify(token_list)
            out_line = _make_words(in_line, iob_list)
            fo.write(' '.join(out_line)+'\n')





