**Bidirectional-LSTM-CRF** model used for NER tasks doesn't reply on language-specific knowledge or resources such as gazetteers. It may achieve state-of-the-art performance compared to other models such as Hidden-Markov-Model.



**References:**
* [Neural Architectures for Named Entity Recognition](https://arxiv.org/pdf/1603.01360.pdf)
* [CRF Layer on the Top of BiLSTM](https://createmomo.github.io/2017/09/12/CRF_Layer_on_the_Top_of_BiLSTM_1/)
* [Maximum Entropy Principle](https://www.youtube.com/watch?v=ynCkUHPEDOI&t=616s)
* [What is Trasition Matrix?](https://www.youtube.com/watch?v=4zg5bNlHZRg&t=20s)
* [Introduction to Conditional Random Fields](http://blog.echen.me/2012/01/03/introduction-to-conditional-random-fields/)
* [anaGo: an NER package based on Keras framework and BLSTM-CRF model](https://github.com/Hironsan/anago)
* [Andrew Ng's 5th course Sequence Models of Deep Learning](deeplearning.ai)

Running ner_blstm_crf.py
```
Loading data...
14041 train sequences
3250 valid sequences

{'words': ['"We', 'started', 'running,', 'many', 'of', 'the', 'girls', 'were', 'screaming,"', 'Fatima', 'said.', '"We', 'were', 'running', 'towards', 'the', 'gate.'], 'entities': [{'text': '"We', 'type': 'PER', 'score': 1.0, 'beginOffset': 0, 'endOffset': 1}, {'text': 'Fatima', 'type': 'PER', 'score': 1.0, 'beginOffset': 9, 'endOffset': 10}, {'text': '"We', 'type': 'ORG', 'score': 1.0, 'beginOffset': 11, 'endOffset': 12}]}
```