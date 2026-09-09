> "roof age greater than 10"


### Embedding Matrix (5 × 30)
Each token is mapped to a 30-dimensional vector.

Below is a simplified diagram of the 5 × 30 embedding matrix:

```
Token          Embedding Vector (30 dimensions)
──────────────────────────────────────────────────────────────────────────────
roof     →  [ 0.21, -0.45,  0.12,  0.67, -0.33,  0.08, ...,  0.19 ]   ← 30 numbers
age      →  [-0.11,  0.34,  0.56, -0.22,  0.41, -0.17, ..., -0.05 ]
greater  →  [ 0.48, -0.09, -0.31,  0.25,  0.63,  0.14, ...,  0.37 ]
than     →  [-0.27,  0.52,  0.18, -0.44, -0.12,  0.29, ...,  0.08 ]
10       →  [ 0.15, -0.38,  0.47,  0.09, -0.26,  0.51, ..., -0.22 ]
```


### With Padding (example)
Suppose the model expects sequences of length 8.

We pad the remaining 3 positions with the padding token (usually ID = 0).

```python
Position | Token     | Embedding Vector (30-dim)
---------|-----------|------------------------------------------
1        | roof      | [ 0.21, -0.45,  0.12, ...,  0.19 ]
2        | age       | [-0.11,  0.34,  0.56, ..., -0.05 ]
3        | greater   | [ 0.48, -0.09, -0.31, ...,  0.37 ]
4        | than      | [-0.27,  0.52,  0.18, ...,  0.08 ]
5        | 10        | [ 0.15, -0.38,  0.47, ..., -0.22 ]
6        | <PAD>     | [ 0.00,  0.00,  0.00, ...,  0.00 ]   ← padding_idx
7        | <PAD>     | [ 0.00,  0.00,  0.00, ...,  0.00 ]
8        | <PAD>     | [ 0.00,  0.00,  0.00, ...,  0.00 ]
```

