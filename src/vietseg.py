
###############################################################################
# Model application
###############################################################################

import sys
import network
from input2vec import make_vec
import numpy as np

# Replace this with your model's result
JSON = '../var/60hidden-30epochs-10batch-0.5eta-5.0lambda-7window-100shape-0run.net'
net = network.load(JSON)


def get_iob(arr):
    d = {0: 'i', 1: 'o', 2: 'b'}
    n = np.argmax(arr)
    return d[n]

def classify(token_list):
    "Classify a list of token"
    result = []
    sen_vec = make_vec(token_list)
    for x in sen_vec:
        result.append(get_iob(net.feedforward(x)))
    return result


if __name__ == '__main__':
    n_args = len(sys.argv)
    if n_args < 3:
        print '\n'
        print "VietSeg. Version 0.0.1"
        print "=================================="
        print "Usage: python3 vietseg.py <input file> <output file>"
        print "\t* The input file must be in UTF-8 encoding."
        print "\t* The tokenized text will be written in the output file."
        print "\t* Each token will be surrounded by square brackets []." + '\n'
        print "\t* Newlines are assumed to end sentences, so no tokenized word\
            will continue across a newline."
        print '\n'
        exit(1)
    else:
        input_file_name = sys.argv[1]
        output_file_name = sys.argv[2]
        algo = 'mm'
        model_file_name = './model.pkl'


