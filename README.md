VietSeg
=======

A Vietnamese Word Segmentation program using Neural Network

## How to use:

- Download the source code

- Change to `src` folder

- Put a text file into `src` folder for segmenting, here I use `input.txt`

- Run `python3 vietseg.py input.txt output.txt` (Yeah, this program use
Python3, and Python2 won't work on it, you can fix this, of course)

- Now you got `output.txt` which is segmented

## Performance:

The test uses data from [JVnSegmenter][1]. See below.

Notice I calculate precision rate using IOB tagging accuracy, not word 
segmenting accuracy, so BEWARE when comparing with other programs.

Here is the result after 5 cross-validation runs: 

     ===================
     FINAL RESULTS:
     ===================
     Accuracy on CV #0 is 94.34 %
     Accuracy on CV #1 is 94.88 %
     Accuracy on CV #2 is 94.81 %
     Accuracy on CV #3 is 95.09 %
     Accuracy on CV #4 is 94.59 %
     ===================
     Average accuracy: 94.74 %

The program use some random shuffers, so your result may not be the same 
as mine.

## Train the model yourself:

- Get the data (see links below) and put in the `dat` folder

- Change working directory to `src` folder

- Run `python3 word2vec.py` to get vectorized words for our segmenting model 
using Word2Vec library (Word2Vec itself is a neural network)

- Run `python3 learn.py` to really train the segmenting model

- Now you can use `python3 vietseg.py <input file> <output file>` as described above


## Data for training model:

- Vietnamese corpus:
    - File: [VNESEcorpus.txt][2]
    - Move the file to `dat` folder

- Vietnamese IOB training data:
    - File: [trainingdata.tar.gz][3]
    - Untar and put 10 files: test1.iob2 -> test5.iob2, train1.iob2 -> train5.iob2 to
    `dat` folder, along with `VNESEcorpus.txt`

## Future works:

- Speed up the network
- Use a professional deep learning package (Theano, Caffe, etc)
- Train the model with bigger corpus and training data file, like [these][9]
- Deal with uppercase characters
- Build a web app

## Acknowledgment:

This program use some code from [wendykan][4] and [mnielsen][5]. 
View the source code for detail.

## Similar programs:

- [JVnSegmenter][1]: Java
- [vnTokenizer][6]: Java
- [Dongdu][7]: C++
- [VLSP][8]: PHP?

## Last words:

> sophisticated algorithm ≤ simple learning algorithm + good training data 


[1]: http://jvnsegmenter.sourceforge.net/
[2]: http://viet.jnlp.org/download-du-lieu-tu-vung-corpus
[3]: http://sourceforge.net/projects/jvnsegmenter/files/jvnsegmenter/JVnSegmenter/trainingdata.tar.gz/download
[4]: https://github.com/wendykan/DeepLearningMovies
[5]: https://github.com/mnielsen/neural-networks-and-deep-learning
[6]: http://mim.hus.vnu.edu.vn/phuonglh/softwares/vnTokenizer
[7]: https://github.com/rockkhuya/DongDu
[8]: http://vlsp.vietlp.org:8080/demo/?page=seg_pos_chunk
[9]: http://vlsp.vietlp.org:8080/demo/?page=resources




