# Tiny Embeddings

### 1. Declare list of input sentance
```text
sentances = [
    "roof age greater than 10",
     "roof age less than 20",
    "roof age greater than 15",
    "construction date greater than 2020",
    

]
```
### 2. vocabulary building
```python

words = set()

for sentence in sentences:
    words.update(sentence.split())

```

### 3. Word to ID and ID to Word Mapping
```python

    def word_to_token_id_map(self):
        return {word:i for i,word in enumerate(sorted(self.vocabulary))}
    def id_to_word_map(self,data):
        # data -> word_to_token_id_map
        return {i:w for w,i in data.items()}
```
### 4. Create training data

```text
Context → target
eg:
    roof age -> "greater"
================================
Training examples:
['roof', 'age'] → greater
['age', 'greater'] → than
['greater', 'than'] → 10
['roof', 'age'] → less
['age', 'less'] → than
['less', 'than'] → 20
['roof', 'age'] → greater
['age', 'greater'] → than
['greater', 'than'] → 15
['construction', 'date'] → greater
['date', 'greater'] → than
['greater', 'than'] → 2020
```







