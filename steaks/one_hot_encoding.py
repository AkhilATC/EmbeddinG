
import re


text = "i love mountains. i love cats."

# Normalise the text and preserve each word's first-seen order in the vocabulary.
tokens = re.findall(r"\b\w+\b", text.lower())
vocabulary = list(dict.fromkeys(tokens))

# Each vector has one position per vocabulary word; its own position is 1.
one_hot_vectors = {
    word: [int(word == vocabulary_word) for vocabulary_word in vocabulary]
    for word in vocabulary
}

print("Vocabulary:", vocabulary)
print("One-hot vectors:")
for word, vector in one_hot_vectors.items():
    print(f"{word}: {vector}")
