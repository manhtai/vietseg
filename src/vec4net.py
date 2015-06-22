
###############################################################################
# Convert word to vector
###############################################################################

import numpy as np
from gensim.models import Word2Vec

MODEL = Word2Vec.load('100features_10context_7minwords.tmp')
WINDOW = 7
SHAPE = MODEL.syn0.shape[1]

def word2index(model, word):
    "Convert word to index using result from Word2Vec learning"
    try:
        if word == -1:
            result = np.zeros((SHAPE,))
        else:
            result = model[word]
    except:
        np.random.seed(len(word))
        result = 0.2 * np.random.uniform(-1, 1, SHAPE)
    return result

def context_window(l):
    "Make context window for a given word, 'WINDOW' must be odd"
    l = list(l)
    lpadded = WINDOW//2 * [-1] + l + WINDOW//2 * [-1]
    out = [ lpadded[i:i+WINDOW] for i in range(len(l)) ]
    assert len(out) == len(l)
    return out

def context_matrix(model, conwin):
    "Return a list contain map element for each context window of 1 word"
    return [map(lambda x: word2index(model, x), win) for win in conwin]

def context_vector(cm):
    "Convert context matrix to vector"
    return [np.squeeze(np.asarray(list(x))).reshape((WINDOW*SHAPE,1)) for x in cm]

def iob_map(iob):
    "Convert i/o/b to vector"
    d = {'i': [1,0,0], 'o': [0,1,0], 'b': [0,0,1]}
    return np.asarray(d[iob]).reshape((3,1))

def iob_vector(iob):
    return map(iob_map, iob)

def make_tuple(context_vector, iob_vector):
    return list(zip(context_vector, iob_vector))

def make_vec(sen):
    "Make vector from a sentence for feeding to neural net directly"
    cw = context_window(sen)
    cm = context_matrix(MODEL, cw)
    return context_vector(cm)

def make_list(sentences):
    "Make list for neural net learning"
    sen, iob = sentences
    cw = context_window(sen)
    cm = context_matrix(MODEL, cw)
    cv = context_vector(cm)
    iv = iob_vector(iob)
    return make_tuple(cv, iv)

if __name__ == '__main__':
    sen = ['mobifone', 'đầu', 'tư', 'hơn', '2', 'tỉ', 'đồng', 'phát', 'triển', 'mạng']
    iob = ['b', 'b', 'i', 'b', 'b', 'b', 'b', 'b', 'i', 'b']
    sentences = (sen, iob)
    lst = make_list(sentences)
    vec = make_vec(sen)




