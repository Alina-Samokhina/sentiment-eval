# Sentiment analysis

Project for course Deep learning in NLP by DeepPavlov.

Based on http://www.dialog-21.ru/evaluation/2015/sentiment/

**Status**: work in progress.

*Done*: 
 - Aspect extraction

*To do*: 
 - Aspects classification

 - Aspects sentiment

 - Generalization of results

## Main idea
In this project we're trying to do aspect-based sentiment analysis in Russian.
Dataset is restaurant reviews from {}

Our intention is to follow such steps:
 1. Extract aspects from text
 2. Classify them as the're abote the whole restaurant, food or service
 3. Classify sentiment regarding these aspects: negative, positive or neutral
 4. Get overall sentiment of the review regarding categories by summing up steps two and three.

__Input__: reviews about restaurants


__Output__: generalized sentiment on different restaurant parameters

 ## Details on each step


 ### Aspects extraction

 1. Get general embeddings (we have {fast-text skipgram model}) May use conversational RuBERT also
 2. Train specific embeddings on our train dataset (wor2vec)
 3. Use these two embeddings as an input to a NN. Our proposal is to concat them. Output of the NN is a sequence of 0, 1 and 2 to show if the word is 'not an aspect', 'first word of the aspect' and 'second or other word in an aspect'.
 To get this sequence to sequence to work we'll use at first some simple approach like LSTM (and, maybe, complicate it later)
 
 ### Aspects classification
 1. Use the same word embeddings as an input
 2. Use CNN to classify our aspects as 'whole', 'food', 'service'

 ### Aspects sentiment
 1. Using data on aspects get sentiments on each of them
 2. Group sentiment data by category
 3. Decide on the final sentiment regarding each category

## Results

## References
pymorphy2 normalizing russian words
might be used for POS - tagging to try to improve results

Korobov M.: Morphological Analyzer and Generator for Russian and
Ukrainian Languages // Analysis of Images, Social Networks and Texts,
pp 320-332 (2015).