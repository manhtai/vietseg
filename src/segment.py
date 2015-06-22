
###############################################################################
# Model application
###############################################################################


import network
from input2vec import make_vec
import numpy as np

JSON = '60hidden-30epochs-10batch-0.5eta-5.0lambda-7window-100shape-0run.json'
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


if 1:
    sen = ['mobifone', 'đầu', 'tư', 'hơn', '2', 'tỉ', 'đồng', 'phát', 'triển', 'mạng']
    iob = ['b', 'b', 'i', 'b', 'b', 'b', 'b', 'b', 'i', 'b']
    result = classify(sen)




