**References:**
* [TOWARDS AI-COMPLETE QUESTION ANSWERING: A SET OF PREREQUISITE TOY TASKS](https://arxiv.org/pdf/1502.05698.pdf)
* https://www.quora.com/Logical-Reasoning-Whats-the-difference-between-deduction-and-induction
* https://nlp.stanford.edu/projects/coref.shtml

**Glossary**
* **Deductive reasoning** refers to the act of reaching a conclusion by showing that such a conclusion must follow from a set of premises. In contrast, **inductive reasoning** refers to the act of reaching a conclusion by abstracting or generalizing a premise.
* **Coreference resolution** is the task of finding all expressions that refer to the same entity in a text. It is an important step for a lot of higher level NLP tasks that involve natural language understanding such as document summarization, question answering, and information extraction.


**Data Setup**
* A set of training and test data, with the intention that a successful model performs
well on test data. The supervision in the training set is given by the true answers to questions, and the set of relevant statements for answering a given question, which may or may not be used by the learner. Correct answers are limited to a
single word (Q: Where is Mark? A: bathroom), or else a list of words (Q: What is Mark holding?)
as evaluation is then clear-cut, and is measured simply as right or wrong.
* The task is noiseless and an average human able to read that language can potentially achieve 100% accuracy.
* The data itself is produced using a simple simulation of characters and objects moving around and interacting in locations
