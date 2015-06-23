import numpy as np
from vietseg import _make_words, _get_iob
from learn import get_files, get_sentences
from vec4net import make_vec
import network



def _make_hyp_ref(RUN):
    "Make human sentences and machine sentences"
    JSON = '../var/30hidden-30epochs-10batch-0.5eta-5.0lambda-7window-100shape-{}run.net'.format(RUN)
    net = network.load(JSON)
    
    def classify(token_list):
        "Classify a list of token to compare with human's segmentation"
        result = []
        sen_vec = make_vec(token_list)
        for x in sen_vec:
            result.append(_get_iob(net.feedforward(x)))
        return result
    
    test_list  = get_files('test')
    test_sent  = get_sentences(test_list, RUN)
    sents_ref = [_make_words(sent, tags) for sent, tags in test_sent]
    sents_hyp = [_make_words(sent, classify(sent)) for sent, _ in test_sent]
    return sents_ref, sents_hyp

def _run_test(RUN):
    "Test performace on CV #RUN"
    # change RUN to constant for evaluate 1 specific model to all test sets
    # sents_ref, sents_hyp = _make_hyp_ref(3)
    sents_ref, sents_hyp = _make_hyp_ref(RUN)
    n_hyp = 0
    n_corr = 0
    n_ref = 0
    nSents = len(sents_ref)
    n_hyps = []
    n_refs = []
    n_corrs = []
    for n in range(nSents):
        sent1 = sents_ref[n] # Human's segmenting
        sent2 = sents_hyp[n] # Machine's segmenting
        
        n_ref_ = len(sent1)
        n_hyp_ = len(sent2)
        
        # Finding optimal alignment and consequently no. of correct words
        # by dynamic programming. Longest Common Subsequence problem.
        l = np.zeros([n_ref_+1, n_hyp_+1])
        
        for row in range(1,l.shape[0]):
            for col in range(1,l.shape[1]):
                if sent1[row-1] == sent2[col-1]:
                    l[row][col] = l[row-1][col-1] + 1
                else:
                    l[row][col] = max([l[row][col-1], l[row-1][col]])
        n_corr_ = l[n_ref_][n_hyp_]
        
        n_hyp += n_hyp_
        n_ref += n_ref_
        n_corr += n_corr_
        n_hyps.append(n_hyp_)
        n_corrs.append(n_corr_)
        n_refs.append(n_ref_)
    
    prec = n_corr / n_hyp
    recall = n_corr / n_ref
    fratio = 2*prec*recall / (prec + recall)
    
    return prec, recall, fratio

def _main(nRuns):
    "Run the test"
    P_ = R_ = F_ = 0
    print("RESULT:")
    print("===================")
    for RUN in range(nRuns):
        prec, recall, fratio = _run_test(RUN)
        print("Run " + "%.0f" % RUN + ": P = " + "%.4f" % prec + ", R = " + \
            "%.4f" % recall + ", F = " + "%.4f" % fratio)
        
        P_ += prec
        R_ += recall
        F_ += fratio

    P_  /= nRuns
    R_  /= nRuns
    F_  /= nRuns

    print("===================")
    print("Avg.   P = "+ "%.4f" % P_ + ", R = " + "%.4f" % R_ + ", F = " +\
        "%.4f" % F_)

if __name__ == '__main__':
    _main(5)


