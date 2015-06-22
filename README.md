VietSeg
=======

A Vietnamese Word Segmentation program using Neural Network

## How to use:

- Download the source code

- Change to `src` folder

- Put a text file into `src` folder for segmenting, here I use `input.txt`

- Run `python3 segment.py input.txt output.txt`

- Now you got `output.txt` which is segmented

## Performance:

~94% precision on training data set. Somewhat the same as 
[JVnSegmenter][1], but much more smaller.

## Train the model yourself:

- Get the data (see links below) and put in the `dat` folder

- Change working directory to `src` folder

- Run `python3 word2vec.py` to get vectorized words for our segmenting model 
using Word2Vec library (Word2Vec itself is a neural network)

- Run `python3 learn.py` to really train the segmenting model

- Now you can use `python3 segment.py` as described above


## Data for training model

- Vietnamese corpus:
    - File: [VNESEcorpus.txt][2]
    - Extract to `dat` folder

- Vietnamese IOB training data:
    - File: [trainingdata.tar.gz][3]
    - Untar and put 10 files: test1.iob2 -> test5.iob2, train1.iob2 -> train5.iob2 to
    `dat` folder, along with `VNESEcorpus.txt`

## Acknowledgment:

This program use some code from [wendykan][4] and [mnielsen][5]. 
View the source code for detail.


[1]: http://jvnsegmenter.sourceforge.net/
[2]: http://viet.jnlp.org/download-du-lieu-tu-vung-corpus
[3]: http://sourceforge.net/projects/jvnsegmenter/files/jvnsegmenter/JVnSegmenter/trainingdata.tar.gz/download
[4]: https://github.com/wendykan/DeepLearningMovies
[5]: https://github.com/mnielsen/neural-networks-and-deep-learning



