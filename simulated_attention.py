# ----------------------------------------
# What:
# Simulated attention over a sentence to see which word "it" refers to.
# Deterministic vectors + cosine similarity + temperature-scaled softmax
# produce stable, interpretable weights.
# ----------------------------------------
# What to observe:
# - Weights concentrate on semantically plausible words (e.g., "animal").
# - Temperature (tau) sharpens or smooths the distribution.
# ----------------------------------------
# Goal:
# Make attention weights meaningful and reproducible, avoiding randomness.
# ----------------------------------------

import numpy as np

# Step 1: Define the sentence and tokens
tokens = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired"]

# Step 2: Deterministic word vectors (seeded for reproducibility)
np.random.seed(7)  # fix randomness
dim = 4

# Baseline random vectors
base_vectors = {tok: np.random.rand(dim) for tok in tokens}

# Optional nudge: make "animal" closer to "it" semantically (for clarity)
# We craft "it" by nudging toward "animal" so the demo highlights pronoun resolution.
word_vectors = base_vectors.copy()
word_vectors["it"] = (0.7 * base_vectors["animal"] + 0.3 * base_vectors["it"])  # slight bias toward "animal"

# Step 3: Build Q, K, V
query = word_vectors["it"]
keys = np.array([word_vectors[token] for token in tokens])
values = keys.copy()  # basic attention: V = K

# Helper: cosine similarity (more stable than raw dot product)
def cosine_sim(a, b, eps=1e-12):
    a = np.asarray(a); b = np.asarray(b)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (max(na, eps) * max(nb, eps)))

# Step 4: Compute similarity scores (Q·K normalized)
scores = np.array([cosine_sim(query, k) for k in keys])

# Step 5: Temperature-scaled softmax (controls sharpness)
def softmax(x, tau=0.7):
    # subtract max for numerical stability
    z = (x - np.max(x)) / max(tau, 1e-12)
    e = np.exp(z)
    return e / e.sum()

weights = softmax(scores, tau=0.7)

# Step 6: Weighted sum to get attended output
attended_output = np.dot(weights, values)

# Step 7: Show attention weights
print("Attention weights for 'it':")
for token, weight in zip(tokens, weights):
    print(f"{token:10s}: {weight:.3f}")

print("\nScores (cosine similarity):")
for token, s in zip(tokens, scores):
    print(f"{token:10s}: {s:.3f}")

# Optional: Show final attended vector
print("\nAttended output vector:", np.round(attended_output, 3))
