# langmodel
Simple implementation of a word-based language model that learns to predict the next word based on the previous one by training on a provided text dataset (Karpathy's Tiny Shakespeare in this case). 
This model is a rudimentary form of a language model and does not include complex features like embeddings, LSTMs or attention mechanisms found in modern language models. Instead, it provides a simple demonstration of how a language model can be implemented using a transition matrix to capture statistical patterns in the text data.

How it works:
1. Text data  loaded from a file, preprocessed (converting it to lowercase and tokenizing it into words).
2. Input-Output pairs are created from the text data for training the language model.
3. A simple word based language model is defined with a transition matrix representing the probabilities of transitioning from one word to another.
4. The model is trained on the training data.
5. Text generation is performed by predicting the next word based on the current word using the trained model.

Compute perplexity for the language model. It is a level of "certanty" of the Model and is calculated as the inverse probability of the test set normalized by the number of words. 
In this file it is being computed this way:
1. Iterating over the test data;
2. Computing the probability of each word given the preceding words;
3. Accumulating the log probability;
4. Final reuslt is calculated by exponentiating the negative log probability divided by the number of words.

Other way to compute perplexity is 2^(average bits per word) for a given test text set. Perplexity of 1 will mean that we don't need to predict the word because its known
