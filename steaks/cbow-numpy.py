import numpy as np


# --------------------------------------------------
# 1. Training data
# --------------------------------------------------

sentence = "i love machine learning".split()

vocabulary = sorted(set(sentence))

word_to_id = {
    word: i
    for i, word in enumerate(vocabulary)
}

id_to_word = {
    i: word
    for word, i in word_to_id.items()
}

vocab_size = len(vocabulary)

print("Vocabulary:")
print(word_to_id)


# --------------------------------------------------
# 2. Create CBOW training examples
# --------------------------------------------------

window_size = 1

training_data = []

for i in range(window_size, len(sentence) - window_size):

    context = (
        sentence[i - window_size:i]
        + sentence[i + 1:i + window_size + 1]
    )

    target = sentence[i]

    context_ids = [
        word_to_id[word]
        for word in context
    ]

    target_id = word_to_id[target]

    training_data.append((context_ids, target_id))


print("\nTraining data:")

for context, target in training_data:
    print(
        [id_to_word[i] for i in context],
        "->",
        id_to_word[target]
    )


# --------------------------------------------------
# 3. Model parameters
# --------------------------------------------------

embedding_dim = 5

learning_rate = 0.05

epochs = 5000


# --------------------------------------------------
# 4. Initialize weights
# --------------------------------------------------

np.random.seed(42)

# Input → hidden / embedding matrix
#
# Shape:
# vocab_size × embedding_dim
#
# Example:
# 4 × 5

W = np.random.randn(
    vocab_size,
    embedding_dim
) * 0.01


# Hidden → output matrix
#
# Shape:
# embedding_dim × vocab_size

W_out = np.random.randn(
    embedding_dim,
    vocab_size
) * 0.01


# Bias
b = np.zeros(vocab_size)


# --------------------------------------------------
# 5. Softmax
# --------------------------------------------------

def softmax(x):

    # Numerical stability
    x = x - np.max(x)

    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x)


# --------------------------------------------------
# 6. Training
# --------------------------------------------------

for epoch in range(epochs):

    total_loss = 0

    for context_ids, target_id in training_data:

        # ------------------------------------------
        # Forward pass
        # ------------------------------------------

        # Get embeddings of context words
        #
        # Example:
        #
        # context = [i, machine]
        #
        # W[context_ids]
        #
        # gives:
        #
        # [
        #   embedding(i),
        #   embedding(machine)
        # ]

        context_vectors = W[context_ids]


        # CBOW combines context embeddings
        #
        # Here we use average

        hidden = np.mean(
            context_vectors,
            axis=0
        )


        # Hidden → output

        scores = np.dot(
            hidden,
            W_out
        ) + b


        # Convert scores → probabilities

        probabilities = softmax(scores)


        # ------------------------------------------
        # Loss
        # ------------------------------------------

        loss = -np.log(
            probabilities[target_id] + 1e-10
        )

        total_loss += loss


        # ------------------------------------------
        # Backpropagation
        # ------------------------------------------

        # Gradient of softmax + cross entropy

        dscores = probabilities.copy()

        dscores[target_id] -= 1


        # ------------------------------------------
        # Gradient for W_out
        # ------------------------------------------

        dW_out = np.outer(
            hidden,
            dscores
        )


        # Gradient for bias

        db = dscores


        # ------------------------------------------
        # Gradient flowing back to hidden
        # ------------------------------------------

        dhidden = np.dot(
            W_out,
            dscores
        )


        # ------------------------------------------
        # Gradient for context embeddings
        # ------------------------------------------

        # Because:
        #
        # hidden = mean(context_vectors)
        #
        # gradient is divided by number of
        # context words.

        dcontext = (
            dhidden / len(context_ids)
        )


        # ------------------------------------------
        # Update W_out and bias
        # ------------------------------------------

        W_out -= learning_rate * dW_out

        b -= learning_rate * db


        # ------------------------------------------
        # Update embedding matrix W
        # ------------------------------------------

        for word_id in context_ids:

            W[word_id] -= (
                learning_rate * dcontext
            )


    # Print progress

    if epoch % 500 == 0:

        print(
            f"Epoch {epoch}, "
            f"Loss: {total_loss:.4f}"
        )


# --------------------------------------------------
# 7. Display learned embeddings
# --------------------------------------------------

print("\nLearned embeddings:\n")

for word, word_id in word_to_id.items():

    print(
        f"{word:10} -> "
        f"{W[word_id]}"
    )