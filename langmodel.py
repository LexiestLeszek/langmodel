import torch
import random

class SimpleWordLanguageModel:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.transition_matrix = torch.zeros((vocab_size, vocab_size))
    
    def train(self, training_data):
        for input_ix, output_ix in training_data:
            self.transition_matrix[input_ix, output_ix] += 1
    
    def predict_next_word(self, input_word_ix):
        output_probs = self.transition_matrix[input_word_ix] / self.transition_matrix[input_word_ix].sum()
        return torch.multinomial(output_probs, 1).item()
    
    def generate_text(self, num_words, dataset_name, word_to_ix, ix_to_word):
        generated_text = []
        current_word_ix = random.choice(list(word_to_ix.values()))  # Start with a random word
        
        for _ in range(num_words):
            generated_text.append(ix_to_word[current_word_ix])
            current_word_ix = self.predict_next_word(current_word_ix)
        
        print(f"Generated text using the trained model on {dataset_name}:\n")
        print(' '.join(generated_text))

# Load and preprocess the text data
with open('tiny_shakespeare.txt', 'r') as file:
    text = file.read().lower()  # Convert text to lowercase
    words = text.split()  # Tokenize the text into words

# Create word-to-index and index-to-word mappings
word_to_ix = {word: i for i, word in enumerate(set(words))}
ix_to_word = {i: word for word, i in word_to_ix.items()}

# Generate training data by creating input-output pairs
training_data = [(word_to_ix[words[i]], word_to_ix[words[i + 1]]) for i in range(len(words) - 1)]

# Train the simple word-based language model
model = SimpleWordLanguageModel(vocab_size=len(word_to_ix))
model.train(training_data)

# Generate text using the trained model
model.generate_text(100, 'tiny_shakespeare.txt', word_to_ix, ix_to_word)
