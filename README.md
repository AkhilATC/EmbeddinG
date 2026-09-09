# EmbeddinG

A small learning project for experimenting with text embeddings: numerical vector
representations that make semantic similarity, search, and clustering possible.

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

The requirements use PyTorch's CPU wheel index, so no CUDA-capable GPU is
needed for the custom NLU layers.

## Custom NLU embedding layer

`nlu_core.py` contains `CustomEmbeddingLayer`, a CPU PyTorch embedding layer.
Pass token IDs to `.embedding(...)` (or call the layer directly) to get a
trainable vector for every token:

```python
from nlu_core import CustomEmbeddingLayer

embeddings = CustomEmbeddingLayer(vocab_size=10_000, embedding_dim=128)
token_vectors = embeddings.embedding([[4, 12, 9]])
# torch.Size([1, 3, 128])
```

## First experiment

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = ["I enjoy learning about AI", "Machine learning is fascinating"]
embeddings = model.encode(texts)

print(embeddings.shape)
```

## Terminal conversation prototype

Run the local conversational terminal interface with:

```bash
python terminal_agent.py
```

It has `/help`, `/history`, `/clear`, and `/exit` commands. The current
response engine is intentionally local and simple; later, replace
`ConversationalAgent.respond` in `terminal_agent.py` with a RAG implementation.

From here, try comparing embeddings with cosine similarity, building a small
semantic search index, or visualising groups of related text.

## One-hot encoding example

Run the included example with:

```bash
python one_hot_encoding.py
```

For the text `i love mountains. i love cats.`, lowercasing and removing
punctuation produces this token sequence:

```text
["i", "love", "mountains", "i", "love", "cats"]
```

The vocabulary keeps every unique word in its first-seen order:

```text
["i", "love", "mountains", "cats"]
```

Each vector has one position for each vocabulary word. A `1` identifies the
word's position; all other positions are `0`.

| Word | `i` | `love` | `mountains` | `cats` | One-hot vector |
| --- | ---: | ---: | ---: | ---: | --- |
| `i` | 1 | 0 | 0 | 0 | `[1, 0, 0, 0]` |
| `love` | 0 | 1 | 0 | 0 | `[0, 1, 0, 0]` |
| `mountains` | 0 | 0 | 1 | 0 | `[0, 0, 1, 0]` |
| `cats` | 0 | 0 | 0 | 1 | `[0, 0, 0, 1]` |

The repeated occurrences of `i` and `love` use the same vectors above. Unlike
semantic embeddings, one-hot vectors do not capture any relationship between
words; they only identify a word in a fixed vocabulary.

## Notes

- The first model download happens automatically and may take a moment.
- Keep API keys and large downloaded models out of version control.
