import torch
import random
import math

# Compute perplexity for the language model
# It is calculated as the inverse probability of the test set normalized by the number of words.
# 1. Iterating over the test data;
# 2. Computing the probability of each word given the preceding words;
# 3. Accumulating the log probability;
# 4. Final reuslt is calculated by exponentiating the negative log probability divided by the number of words.
# Other way to compute perplexity is 2^(average bits per word) for a given test text set
# Perplexity of 1 will mean that we don't need to predict the word because its known

class SimpleWordLanguageModel:
    def __init__(self):
        self.transition_matrix = None
        self.word_to_ix = None
        self.ix_to_word = None
        self.vocab_size =  0
    
    def load_and_train(self, filename):
        # Load the document
        with open(filename, 'r') as file:
            text = file.read().lower()
        
        # Tokenize the text into words
        words = text.split()
        
        # Create word-to-index and index-to-word mappings
        self.word_to_ix = {word: i for i, word in enumerate(set(words))}
        self.ix_to_word = {i: word for word, i in self.word_to_ix.items()}
        self.vocab_size = len(self.word_to_ix)
        
        # Generate training data by creating input-output pairs
        training_data = [(self.word_to_ix[words[i]], self.word_to_ix[words[i +  1]]) for i in range(len(words) -  1)]
    
        self.transition_matrix = torch.zeros((self.vocab_size, self.vocab_size))
        for input_ix, output_ix in training_data:
            self.transition_matrix[input_ix, output_ix] +=  1
        
        total_log_prob = 0
        num_words = 0
        
        for input_ix, output_ix in training_data:
            transition_probs = model.transition_matrix[input_ix]
            word_prob = transition_probs[output_ix] / transition_probs.sum()
            total_log_prob += math.log(word_prob)
            num_words += 1

        perplexity = math.exp(-total_log_prob / num_words)
        print(f"Training finished! Based on the {filename} dataset the Model Perplexity is: {round(perplexity,4)}\n")
    
    def generate(self, num_words):
        generated_text = []
        current_word_ix = random.choice(list(self.word_to_ix.values()))  # Start with a random word
        
        for _ in range(num_words):
            generated_text.append(self.ix_to_word[current_word_ix])
            output_probs = self.transition_matrix[current_word_ix] / self.transition_matrix[current_word_ix].sum()
            current_word_ix = torch.multinomial(output_probs,  1).item()
        
        print("Generating:\n")
        print(' '.join(generated_text))

# Use the refactored model
model = SimpleWordLanguageModel()
model.load_and_train('tiny_shakespeare.txt')
model.generate(100)
#model.train_and_generate('tiny_shakespeare.txt',  100)