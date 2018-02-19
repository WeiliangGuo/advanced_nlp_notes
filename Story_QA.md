**References:**
* [01.TOWARDS AI-COMPLETE QUESTION ANSWERING: A SET OF PREREQUISITE TOY TASKS](https://arxiv.org/pdf/1502.05698.pdf)
* [02.End-To-End Memory Networks](https://arxiv.org/pdf/1503.08895.pdf)
* https://www.quora.com/Logical-Reasoning-Whats-the-difference-between-deduction-and-induction
* https://nlp.stanford.edu/projects/coref.shtml

**Glossary**
* **Deductive reasoning** refers to the act of reaching a conclusion by showing that such a conclusion must follow from a set of premises. In contrast, **inductive reasoning** refers to the act of reaching a conclusion by abstracting or generalizing a premise.
* **Coreference resolution** is the task of finding all expressions that refer to the same entity in a text. It is an important step for a lot of higher level NLP tasks that involve natural language understanding such as document summarization, question answering, and information extraction.
* **End-to-End** in deep learning literatures, it means training a model almost from raw training data  directly to target values without other intermidiary procedures(e.g. coreference and semantic role labeling). This is a promising approach if huge amounts of training data available. 

**Problem Description**
* According to [01], any system aiming at full text understanding and reasoning must perform well on all the 20 types of tasks they suggested. these tasks should be trained seperately rather jointly using end-to-end memory networks(MemN2N). 
* Examples tasks: 
  * **Task 1: Single Supporting Fact**
    * **Story**: * Mary went to the bathroom. * John moved to the hallway. * Mary travelled to the office.
    * **Question**: Where is Mary? 
    * **Answer**: office
  * **Task 3: Three Supporting Facts**
    * **Story**: * John picked up the apple. * John went to the office. * John went to the kitchen. * John dropped the apple.
    * **Question**: Where was the apple before the kitchen? 
    * **Answer**: office  
  * **Task 19: Path Finding Task 20: Agent’s Motivations**
    * **Story**: * The kitchen is north of the hallway.
                 * The bathroom is west of the bedroom.
                 * The den is east of the hallway. 
                 * The office is south of the bedroom.
    * **Q1**: How do you go from den to kitchen? 
    * **A1**: west, north 
    * **Q2**: How do you go from office to bathroom?
    * **A2**: north, west
    
**Data Setup**
* A set of training and test data, with the intention that a successful model performs
well on test data. The supervision in the training set is given by the true answers to questions, and the set of relevant statements for answering a given question, which may or may not be used by the learner. Correct answers are limited to a
single word (Q: Where is Mark? A: bathroom), or else a list of words (Q: What is Mark holding?)
as evaluation is then clear-cut, and is measured simply as right or wrong.
* The task is noiseless and an average human able to read that language can potentially achieve 100% accuracy.
* The data itself is produced using a simple simulation of characters and objects moving around and interacting in locations

**System Design**
* We use memory networks. They work by a “controller” neural network performing inference over the stored memories that consist of the previous statements in the story. The original proposed model(Weston et al., 2014) performs 2 hops of inference: finding the first supporting fact with the maximum match score with the question, and then the second supporting fact with the maximum match score with both the question and the first fact that was found. The matching function consists of mapping the bag-ofwords for the question and facts into an embedding space by summing word embeddings. The word embeddings are learnt using strong supervision to optimize the QA task. After finding supporting facts, a final ranking is performed to rank possible responses (answer words) given those facts.
* Some extensions to this model：
  * **Adaptive memories** performing a variable number of hops rather than 2, the model is trained to predict a hop or the special “STOP” class. A similar procedure can be applied to output multiple tokens as well.
  * **N-grams** We tried using a bag of 3-grams rather than a bag-of-words to represent the text. In both cases the first step of the MemNN is to convert these into vectorial embeddings.
  * **Nonlinearity** We apply a classical 2-layer neural network with tanh nonlinearity in the matching function.

**Output after running Story_QA.py**
```
Number of training stories: 10000
-
Here's what a "story" tuple looks like (input, query, answer):
(['Mary', 'moved', 'to', 'the', 'bathroom', '.', 'John', 'went', 'to', 'the', 'hallway', '.'], ['Where', 'is', 'Mary', '?'], 'bathroom')
Number of test stories: 1000
-
Vocab size: 22 unique words
Story max length: 68 words
Query max length: 4 words
-
inputs: integer tensor of shape (samples, max_length)
inputs_train shape: (10000, 68)
-
queries: integer tensor of shape (samples, max_length)
queries_train shape: (10000, 4)
-
answers: binary (1 or 0) tensor of shape (samples, vocab_size)
answers_train shape: (10000,)
inputs_test shape: (1000, 68)
queries_test shape: (1000, 4)
answers_test shape: (1000,)
-
Compiling...
Train on 10000 samples, validate on 1000 samples

==========Using rmsprop

Epoch 120/120
   32/10000 [..............................] - ETA: 1s - loss: 0.1411 - acc: 0.9062
  352/10000 [>.............................] - ETA: 1s - loss: 0.0776 - acc: 0.9688
  ......
  ......
 8512/10000 [========================>.....] - ETA: 0s - loss: 0.1029 - acc: 0.9646
 8768/10000 [=========================>....] - ETA: 0s - loss: 0.1031 - acc: 0.9645
 8992/10000 [=========================>....] - ETA: 0s - loss: 0.1036 - acc: 0.9645
 9216/10000 [==========================>...] - ETA: 0s - loss: 0.1034 - acc: 0.9647
 9440/10000 [===========================>..] - ETA: 0s - loss: 0.1043 - acc: 0.9646
 9664/10000 [===========================>..] - ETA: 0s - loss: 0.1049 - acc: 0.9646
 9888/10000 [============================>.] - ETA: 0s - loss: 0.1042 - acc: 0.9647
10000/10000 [==============================] - 2s 246us/step - loss: 0.1047 - acc: 0.9644 - val_loss: 0.2100 - val_acc: 0.9290

==========Using adam

   32/10000 [..............................] - ETA: 2s - loss: 0.0349 - acc: 1.0000
  256/10000 [..............................] - ETA: 2s - loss: 0.0380 - acc: 0.9805
  480/10000 [>.............................] - ETA: 2s - loss: 0.0293 - acc: 0.9875
  704/10000 [=>............................] - ETA: 2s - loss: 0.0292 - acc: 0.9886
  928/10000 [=>............................] - ETA: 2s - loss: 0.0280 - acc: 0.9892
 1152/10000 [==>...........................] - ETA: 2s - loss: 0.0270 - acc: 0.9905
  ......
  ......
 8992/10000 [=========================>....] - ETA: 0s - loss: 0.0418 - acc: 0.9859
 9216/10000 [==========================>...] - ETA: 0s - loss: 0.0409 - acc: 0.9862
 9440/10000 [===========================>..] - ETA: 0s - loss: 0.0403 - acc: 0.9864
 9664/10000 [===========================>..] - ETA: 0s - loss: 0.0401 - acc: 0.9864
 9888/10000 [============================>.] - ETA: 0s - loss: 0.0403 - acc: 0.9865
10000/10000 [==============================] - 2s 249us/step - loss: 0.0407 - acc: 0.9863 - val_loss: 0.0936 - val_acc: 0.9720
```
