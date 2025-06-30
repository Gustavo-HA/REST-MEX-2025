# REST-MEX 2025 – CIMAT-CC Submission

This repository contains the code and models used in the participation of the **CIMAT-CC** team at the [REST-MEX 2025](https://sites.google.com/view/rest-mex/) shared task, part of **IberLEF@SEPLN 2025**. The competition focuses on sentiment analysis and text classification in Spanish-language tourism reviews related to Mexican **Pueblos Mágicos**.

## Tasks Addressed

We tackled three NLP classification subtasks:

1. **Sentiment Polarity Classification (Polarity):** Classifying review sentiment on a 1–5 star scale.
2. **Destination Type Classification (Type):** Classifying whether the review refers to a hotel, restaurant, or attraction.
3. **Magical Town Recognition (Town):** Identifying the specific Pueblo Mágico being referenced.

##  System Overview

- **Polarity & Town:** Fine-tuned [BETO](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased), a BERT model pretrained on Spanish.
- **Type:** Word2Vec + Multilayer Perceptron (MLP) implemented using `gensim` and `scikit-learn`.

We chose a hybrid approach to balance the complexity and requirements of each task.


