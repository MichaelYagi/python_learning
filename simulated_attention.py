# Simulated attention mechanism for a sentence
import numpy as np

# Step 1: Define the sentence and tokenize
tokens = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired"]

# Step 2: Create dummy word vectors (random for illustration)
# np.random.seed(42)
word_vectors = {token: np.random.rand(4) for token in tokens}  # 4D vectors

# Step 3: Define query (Q), keys (K), and values (V)
query = word_vectors["it"]
keys = np.array([word_vectors[token] for token in tokens])
values = keys.copy()  # In basic attention, V = K

# Step 4: Compute dot-product attention scores
scores = np.dot(keys, query)

# Step 5: Apply softmax to get attention weights
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

weights = softmax(scores)

# Step 6: Compute weighted sum of values
attended_output = np.dot(weights, values)

# Step 7: Show attention weights
print("Attention weights for 'it':")
for token, weight in zip(tokens, weights):
    print(f"{token:10s}: {weight:.3f}")

# Optional: Show final attended vector
print("\nAttended output vector:", attended_output.round(3))
