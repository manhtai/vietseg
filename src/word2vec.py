
###############################################################################
# Training Word2Vec to get word embedding vector
###############################################################################

import json
import re
import logging
from gensim.models import Word2Vec
from bs4 import BeautifulSoup


def strip_tags(html):
    "Strip html tags"
    return BeautifulSoup(html).get_text(' ')

def get_stops():
    "Get stop words from files"
    return json.load(open('stops.json'))

def text_to_token(text, remove_stopwords=False):
    "Get list of token for training"
    # Strip HTML
    text = strip_tags(text)
    # Keep only word
    text = re.sub("\W", " ", text)
    # Lower and split sentence
    token = text.lower().split()
    # Don't remember the number
    for i in range(len(token)):
        token[i] = len(token[i])*'DIGIT' if token[i].isdigit() else token[i]
    # Remove stopwords if must
    if remove_stopwords:
        stops = set(get_stops())
        token = [w for w in token if not w in stops]
    return token

def read_sentences(fp='../data/VNESEcorpus.txt.tmp'):
    "Read and split token from text file"
    sentences = []
    with open(fp, 'r') as f:
        for line in f:
            if '|' not in line: # Remove menu items
                sentences.append(text_to_token(line.strip()))
    return sentences

if __name__ == '__main__':
    # Read data from files
    sentences = read_sentences()
    print('Loaded {} sentences!'.format(len(sentences)))
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)
    # Set values for various parameters
    num_features = 100    # Word vector dimensionality
    min_word_count = 7    # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words
    print("Training Word2Vec model...")
    # Initialize and train the model
    model = Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling, seed=1)
    model.init_sims(replace=True)
    model_name = "{}features_{}context_{}minwords.tmp".format(
        num_features, context, min_word_count
    )
    model.save(model_name)




