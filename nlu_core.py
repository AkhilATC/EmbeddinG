
"""Core neural-network building blocks for the local NLU pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.optim as optim



class CustomEmbeddingLayer(nn.Module):
    """Convert token IDs into trainable dense vectors on the CPU.

    Parameters
    ----------
    vocab_size:
        Number of tokens in the tokenizer vocabulary.
    embedding_dim:
        Number of features produced for each token.
    padding_idx:
        Optional token ID whose vector remains zero during training.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

    def embedding(self, token_ids: Tensor | Sequence[int] | Sequence[Sequence[int]]) -> Tensor:
        """Return a vector for every token ID.

        ``token_ids`` may be one sequence (``[4, 12, 9]``) or a batch of
        sequences (``[[4, 12], [9, 0]]``). The returned shape is the input
        shape followed by ``embedding_dim``.
        """
        ids = torch.as_tensor(token_ids, dtype=torch.long, device="cpu")
        return self.embedding_layer(ids)

    def forward(self, token_ids: Tensor | Sequence[int] | Sequence[Sequence[int]]) -> Tensor:
        """Make the layer callable in normal PyTorch models."""
        return self.embedding(token_ids)



# -----------------------------------
# 4. Tiny embedding model
# -----------------------------------

class TinyEmbeddingModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.linear = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, x):

        # x shape:
        #
        # [batch, context_size]
        #
        embeddings = self.embedding(x)

        # [batch, context_size, 32]

        # Combine context embeddings
        embeddings = embeddings.mean(dim=1)

        # [batch, 32]

        output = self.linear(embeddings)

        # [batch, vocab_size]

        return output


class PreprocessorEmbeddingPrior:
    def __init__(self):
        self.sentences = [
            "roof age greater than 10",
            "roof age less than 20",
            "roof age greater than 15",
            "construction date greater than 2020"
        ]
        self.vocabulary = set()

    def build_vocabulary(self):

        for word_token in self.sentences:
            self.vocabulary.update(word_token.split())
        print(self.vocabulary)

    def word_to_token_id_map(self):
        return {word:i for i,word in enumerate(sorted(self.vocabulary))}
    def id_to_word_map(self,data):
        return {i:w for w,i in data.items()}

    def training(self,context_size=2):
        word_to_id = self.word_to_token_id_map()
        id_to_word = self.id_to_word_map(word_to_id)
        training_data = []

        for sentence in self.sentences:

            tokens = sentence.split()

            for i in range(context_size, len(tokens)):
                context = tokens[i - context_size:i]
                target = tokens[i]

                context_ids = [
                    word_to_id[word]
                    for word in context
                ]

                target_id = word_to_id[target]

                training_data.append(
                    (context_ids, target_id)
                )

        print("\nTraining examples:")

        for x, y in training_data:
            print(
                [id_to_word[i] for i in x],
                "→",
                id_to_word[y]
            )
        # -----------------------------------
        # 5. Create model
        # -----------------------------------

        embedding_dim = 32

        model = TinyEmbeddingModel(
            len(self.vocabulary),
            embedding_dim
        )

        print("\nModel:")
        print(model)

        # -----------------------------------
        # 6. Loss + optimizer
        # -----------------------------------

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            model.parameters(),
            lr=0.01
        )

        # -----------------------------------
        # 7. Training
        # -----------------------------------

        for epoch in range(1000):

            total_loss = 0

            for context, target in training_data:
                x = torch.tensor(
                    [context],
                    dtype=torch.long
                )

                y = torch.tensor(
                    [target],
                    dtype=torch.long
                )

                # Forward
                output = model(x)

                # Loss
                loss = criterion(output, y)

                # Backpropagation
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                total_loss += loss.item()

            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}, "
                    f"Loss = {total_loss:.4f}"
                )

        # EMBEDDINGS SHOWCASING
        embeddings = model.embedding.weight.detach()

        print("\nEmbedding matrix shape:")
        print(embeddings.shape)

        # -----------------------------------
        # 9. Show one word's embedding
        # -----------------------------------

        word = "roof"

        word_id = word_to_id[word]

        roof_vector = embeddings[word_id]

        print("\nEmbedding for:", word)

        print(roof_vector)

        print("Dimension:", len(roof_vector))

if __name__ == "__main__":
    pep = PreprocessorEmbeddingPrior()
    pep.build_vocabulary()
    pep.training()