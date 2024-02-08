import torch
import random

with open('tiny_shakespeare.txt', 'r') as file:
    text = file.read().lower()  
    words = text.split() 

word_to_ix = {word: i for i, word in enumerate(set(words))}
ix_to_word = {i: word for word, i in word_to_ix.items()}

training_data = [(word_to_ix[words[i]], word_to_ix[words[i + 1]]) for i in range(len(words) - 1)]

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

model = SimpleWordLanguageModel(vocab_size=len(word_to_ix))
model.train(training_data)

generated_text = []
current_word_ix = random.choice(list(word_to_ix.values()))

for _ in range(100):
    generated_text.append(ix_to_word[current_word_ix])
    current_word_ix = model.predict_next_word(current_word_ix)

print(' '.join(generated_text))
