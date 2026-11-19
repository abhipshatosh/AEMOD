# AEMOD

## Embedded vector database

`vector_db` is a small, dependency-free Python vector store. It persists vectors
and JSON metadata in SQLite and performs exact cosine or Euclidean nearest
neighbor search. Each collection uses a consistent vector dimension.

```python
from vector_db import VectorDatabase

with VectorDatabase("data/vectors.sqlite3") as db:
    db.upsert("articles", "article-1", [0.1, 0.2, 0.3], {"title": "Example"})
    matches = db.search("articles", [0.1, 0.2, 0.3], limit=5)

    for match in matches:
        print(match.id, match.distance, match.metadata)
```

`upsert` creates a collection on its first vector and replaces an existing
record with the same ID. `get` retrieves a record, and `delete` removes it.
Search results are ordered by ascending distance; cosine distance is `1 -
cosine_similarity`. Search is exact and scans the selected collection, making
this implementation suitable for small to medium datasets.

Run the test suite with:

```sh
python -m unittest discover -s tests
```