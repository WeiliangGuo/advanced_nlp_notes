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
  
  
