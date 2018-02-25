**Bidirectional-LSTM-CRF** model used for NER tasks doesn't reply on language-specific knowledge or resources such as gazetteers. It may achieve state-of-the-art performance compared to other models such as Hidden-Markov-Model.

**_X = (x<sub>1</sub>; <sub>x2</sub>; : : : ; x<sub>n</sub>)_** is an input sentence of words **_x<sub>i</sub>_**.

**_y = (y<sub>1</sub>; y<sub>2</sub>; : : : ; y<sub>n</sub>)_** is a sequence of predicted tags **_y<sub>i</sub>_** corresponding to each word in above sentence. 

Each such predicted tag is associated with a score: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;s(X;&space;y)&space;=&space;\sum_{i=0}^{n}A_{y_{i},y_{i&plus;1}}&space;&plus;&space;\sum_{i=1}^{n}P_{i,y_{i}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;s(X;&space;y)&space;=&space;\sum_{i=0}^{n}A_{y_{i},y_{i&plus;1}}&space;&plus;&space;\sum_{i=1}^{n}P_{i,y_{i}}" title="s(X; y) = \sum_{i=0}^{n}A_{y_{i},y_{i+1}} + \sum_{i=1}^{n}P_{i,y_{i}}" /></a> , it's a combination of two matrices: **_A_** and **_P_**.

 **_A_** is a matrix of transition[3] probabilities such that **_A<sub>i;j</sub>_** represents the probability of a transition from the
tag **_i_** to tag **_j_**. **_y<sub>0</sub>_** and **_y<sub>n</sub>_** are the start and end tags of a sentence, that we add to the set of possible tags. A is therefore a square matrix of size **_k+2_**(a transition matrix is always a square matrix).

the **_P<sub>i,y</sub>'s_** are the scores associated with each tagging decision for each token which are defined to be the dot product between the embedding of a word-in-context computed with a bidirectional LSTM.

Shape of **_P_** is <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;n*k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;n*k" title="n*k" /></a>, where **_n_** is the number of words in the sentence, **_k_** is the number of distinct NER tags.

In [1], ths scores of transition matrix **A** are acquired from feature functions which were described in [4].

**References:**
* [1. Neural Architectures for Named Entity Recognition](https://arxiv.org/pdf/1603.01360.pdf)
* [2. Maximum Entropy Principle](https://www.youtube.com/watch?v=ynCkUHPEDOI&t=616s)
* [3. What is Trasition Matrix?](https://www.youtube.com/watch?v=4zg5bNlHZRg&t=20s)
* [4. Introduction to Conditional Random Fields](http://blog.echen.me/2012/01/03/introduction-to-conditional-random-fields/)
* [5. anaGo: an NER package based on Keras framework and BLSTM-CRF model](https://github.com/Hironsan/anago)
*  6. Andrew Ng's 5th course Sequence Models of Deep Learning
