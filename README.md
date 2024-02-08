# langmodel
Simple implementation of a word-based language model that learns to predict the next word based on the previous one by training on a provided text dataset (Karpathy's Tiny Shakespeare):

How it works:
1. Text data  loaded from a file, preprocessed (converting it to lowercase and tokenizing it into words).
2. Input-Output pairs are created from the text data for training the language model.
3. A simple word based language model is defined with a transition matrix representing the probabilities of transitioning from one word to another.
4. The model is trained on the training data.
5. Text generation is performed by predicting the next word based on the current word using the trained model.
