import torch
import torch.optim as optim
import numpy as np
import json
from collections import defaultdict
import math

def tokenize(text):
    """A simple tokenizer that lowercases text and splits on whitespace."""
    return text.lower().split()

# ----------------------------
# Step 1: Load the JSON File and Prepare the Corpus
# ----------------------------
# full.json is assumed to be a JSON file containing a list of objects with "question" and "answer" keys.
with open('DB\TOFU\train\full.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Build the corpus: treat each question and answer as separate sentences.
corpus = []
for pair in data:
    corpus.append(pair['question'])
    corpus.append(pair['answer'])

print("Number of sentences in corpus:", len(corpus))

# ----------------------------
# Step 2: Build Vocabulary and the Co-occurrence Matrix
# ----------------------------

# Build vocabulary from the entire corpus.
vocab = set()
for sentence in corpus:
    words = tokenize(sentence)
    vocab.update(words)
vocab = list(vocab)
word_to_idx = {word: i for i, word in enumerate(vocab)}
vocab_size = len(vocab)
print("Vocabulary size:", vocab_size)

# Build a co-occurrence dictionary using a fixed context window.
window_size = 2  # You can adjust the window size as needed.
cooccurrence = defaultdict(float)

for sentence in corpus:
    words = tokenize(sentence)
    for i, word in enumerate(words):
        word_idx = word_to_idx[word]
        # Determine the boundaries for the context window.
        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)
        for j in range(start, end):
            if j == i:
                continue  # Skip the target word itself.
            context_word = words[j]
            context_idx = word_to_idx[context_word]
            # Optionally weight the count by the inverse of the distance.
            distance = abs(i - j)
            increment = 1.0 / distance
            cooccurrence[(word_idx, context_idx)] += increment

# Convert co-occurrence dictionary into a list for vectorized processing.
data_tuples = [(i, j, count) for (i, j), count in cooccurrence.items()]
num_pairs = len(data_tuples)
print("Number of nonzero co-occurrence pairs:", num_pairs)

# Prepare tensors for the nonzero co-occurrence entries.
i_indices = torch.LongTensor([d[0] for d in data_tuples])
j_indices = torch.LongTensor([d[1] for d in data_tuples])
counts = torch.FloatTensor([d[2] for d in data_tuples])

# ----------------------------
# Step 3: Initialize GloVe Model Parameters
# ----------------------------

# Hyperparameters for GloVe.
embedding_dim = 50   # Dimensionality of word embeddings.
x_max = 100.0        # Parameter for the weighting function f(x).
alpha = 0.75         # Exponent for f(x).
learning_rate = 0.05
num_epochs = 100

# Initialize embeddings and biases for target words and context words.
W = torch.randn(vocab_size, embedding_dim, requires_grad=True)       # Target word embeddings.
W_tilde = torch.randn(vocab_size, embedding_dim, requires_grad=True)   # Context word embeddings.
b = torch.randn(vocab_size, requires_grad=True)                        # Target biases.
b_tilde = torch.randn(vocab_size, requires_grad=True)                  # Context biases.

optimizer = optim.Adam([W, W_tilde, b, b_tilde], lr=learning_rate)

# ----------------------------
# Step 4: Train the GloVe Model
# ----------------------------
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Retrieve embeddings and biases for the current co-occurrence pairs.
    w_i = W[i_indices]           # (num_pairs, embedding_dim)
    w_j = W_tilde[j_indices]     # (num_pairs, embedding_dim)
    b_i = b[i_indices]           # (num_pairs,)
    b_j = b_tilde[j_indices]     # (num_pairs,)
    
    # Compute the dot product and add biases.
    dot_product = torch.sum(w_i * w_j, dim=1)  # (num_pairs,)
    prediction = dot_product + b_i + b_j
    
    # Target value is the logarithm of the co-occurrence count.
    log_cooccurrence = torch.log(counts)
    
    # Calculate the squared error term.
    diff = prediction - log_cooccurrence
    
    # Compute the weighting function f(x) for each pair.
    weights = torch.pow(counts / x_max, alpha)
    weights = torch.min(weights, torch.ones_like(weights))
    
    # Compute the total weighted squared error loss.
    loss = torch.sum(weights * (diff ** 2))
    
    # Backpropagate and update parameters.
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}")

# ----------------------------
# Step 5: Combine and Inspect Learned Word Embeddings
# ----------------------------

# It is common practice to combine the target and context embeddings.
final_embeddings = W + W_tilde

print("\nFinal word embeddings (first 5 dimensions printed):")
for word, idx in word_to_idx.items():
    embedding = final_embeddings[idx].detach().numpy()
    print(f"{word}: {embedding[:5]} ...")
