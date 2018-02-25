**Bidirectional-LSTM-CRF** model used for NER tasks doesn't reply on language-specific knowledge or resources such as gazetteers. It may achieve state-of-the-art performance compared to other models such as Hidden-Markov-Model.

**_X = (x<sub>1</sub>; <sub>x2</sub>; : : : ; x<sub>n</sub>)_** is an input sentence of words **_x<sub>i</sub>_**.

**_y = (y<sub>1</sub>; y<sub>2</sub>; : : : ; y<sub>n</sub>)_** is a sequence of predicted tags **_y<sub>i</sub>_** corresponding to each word in above sentence. Each such predicted tag is associated with a score:

<a href="https://www.codecogs.com/eqnedit.php?latex=s(X;&space;y)&space;=&space;\sum_{i=0}^{n}A_{y_{i},y_{i&plus;1}}&space;&plus;&space;\sum_{i=1}^{n}P_{i,y_{i}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s(X;&space;y)&space;=&space;\sum_{i=0}^{n}A_{y_{i},y_{i&plus;1}}&space;&plus;&space;\sum_{i=1}^{n}P_{i,y_{i}}" title="s(X; y) = \sum_{i=0}^{n}A_{y_{i},y_{i+1}} + \sum_{i=1}^{n}P_{i,y_{i}}" /></a>

In [1], ths scores of transition matrix **A** are acquired from feature functions which were described in [2].

Conditional Random Fields are Maximum Entropy Models.Their objective function is based on maximizing the entropy of the model given the constraints in the form of expectation conditions.

**References:**
* [1. Neural Architectures for Named Entity Recognition](https://arxiv.org/pdf/1603.01360.pdf)
* [Maximum Entropy Principle](https://www.youtube.com/watch?v=ynCkUHPEDOI&t=616s)
* https://www.researchgate.net/post/Conditional_Random_fields_vs_Maximum_Entropy_Models
* [What is Trasition Matrix?](https://www.youtube.com/watch?v=4zg5bNlHZRg&t=20s)
* [2. Introduction to Conditional Random Fields](http://blog.echen.me/2012/01/03/introduction-to-conditional-random-fields/)
* [anaGo: an NER package based on Keras framework and BLSTM-CRF model](https://github.com/Hironsan/anago)
* Andrew Ng's 5th course Sequence Models of Deep Learning
