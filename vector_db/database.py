import json
import math
import numbers
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence


@dataclass(frozen=True)
class VectorRecord:
    id: str
    vector: tuple[float, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SearchResult:
    id: str
    distance: float
    metadata: dict[str, Any]


class VectorDatabase:
    """A persistent, exact-search vector database backed by SQLite."""

    def __init__(self, path: str | Path = "vectors.sqlite3") -> None:
        db_path = str(path)
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(
            db_path, check_same_thread=False, isolation_level=None
        )
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.RLock()
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS collections (
                    name TEXT PRIMARY KEY,
                    dimension INTEGER NOT NULL CHECK (dimension > 0)
                );
                CREATE TABLE IF NOT EXISTS vectors (
                    collection TEXT NOT NULL,
                    id TEXT NOT NULL,
                    vector TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    PRIMARY KEY (collection, id),
                    FOREIGN KEY (collection) REFERENCES collections(name)
                        ON DELETE CASCADE
                );
                """
            )

    def upsert(
        self,
        collection: str,
        id: str,
        vector: Sequence[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or replace a vector. The collection dimension is set on first insert."""
        self._validate_name(collection, "collection")
        self._validate_name(id, "id")
        values = self._vector(vector)
        metadata_json = self._metadata_json(metadata)

        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT dimension FROM collections WHERE name = ?", (collection,)
                ).fetchone()
                if row is None:
                    self._connection.execute(
                        "INSERT INTO collections (name, dimension) VALUES (?, ?)",
                        (collection, len(values)),
                    )
                elif row[0] != len(values):
                    raise ValueError(
                        f"collection {collection!r} has dimension {row[0]}, "
                        f"not {len(values)}"
                    )

                self._connection.execute(
                    """
                    INSERT INTO vectors (collection, id, vector, metadata)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (collection, id) DO UPDATE SET
                        vector = excluded.vector,
                        metadata = excluded.metadata
                    """,
                    (collection, id, json.dumps(values), metadata_json),
                )
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def get(self, collection: str, id: str) -> VectorRecord | None:
        """Return a stored vector, or None if the ID does not exist."""
        with self._lock:
            row = self._connection.execute(
                """
                SELECT vector, metadata FROM vectors
                WHERE collection = ? AND id = ?
                """,
                (collection, id),
            ).fetchone()
        if row is None:
            return None
        return VectorRecord(id, tuple(json.loads(row[0])), json.loads(row[1]))

    def delete(self, collection: str, id: str) -> bool:
        """Delete a vector and return whether it existed."""
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM vectors WHERE collection = ? AND id = ?",
                (collection, id),
            )
        return cursor.rowcount > 0

    def search(
        self,
        collection: str,
        query: Sequence[float],
        limit: int = 10,
        metric: Literal["cosine", "euclidean"] = "cosine",
    ) -> list[SearchResult]:
        """Return the nearest vectors, ordered by ascending distance."""
        self._validate_name(collection, "collection")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        if metric not in ("cosine", "euclidean"):
            raise ValueError("metric must be 'cosine' or 'euclidean'")
        query_values = self._vector(query)

        with self._lock:
            row = self._connection.execute(
                "SELECT dimension FROM collections WHERE name = ?", (collection,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown collection: {collection!r}")
            if row[0] != len(query_values):
                raise ValueError(
                    f"collection {collection!r} has dimension {row[0]}, "
                    f"not {len(query_values)}"
                )
            rows = self._connection.execute(
                "SELECT id, vector, metadata FROM vectors WHERE collection = ?",
                (collection,),
            ).fetchall()

        results = []
        query_norm = math.sqrt(sum(value * value for value in query_values))
        for id, vector_json, metadata_json in rows:
            candidate = json.loads(vector_json)
            if metric == "euclidean":
                distance = math.sqrt(
                    sum((a - b) ** 2 for a, b in zip(query_values, candidate))
                )
            else:
                candidate_norm = math.sqrt(sum(value * value for value in candidate))
                if query_norm == 0 or candidate_norm == 0:
                    distance = 1.0
                else:
                    similarity = sum(
                        a * b for a, b in zip(query_values, candidate)
                    ) / (query_norm * candidate_norm)
                    distance = 1.0 - max(-1.0, min(1.0, similarity))
            results.append(
                SearchResult(id, distance, json.loads(metadata_json))
            )

        results.sort(key=lambda result: (result.distance, result.id))
        return results[:limit]

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "VectorDatabase":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    @staticmethod
    def _validate_name(value: str, label: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a non-empty string")

    @staticmethod
    def _vector(vector: Sequence[float]) -> tuple[float, ...]:
        if isinstance(vector, (str, bytes)):
            raise ValueError("vector must be a non-empty sequence of numbers")
        try:
            values = tuple(vector)
        except TypeError as error:
            raise ValueError("vector must be a non-empty sequence of numbers") from error
        if not values:
            raise ValueError("vector must not be empty")
        if any(
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("vector values must be finite numbers")
        return tuple(float(value) for value in values)

    @staticmethod
    def _metadata_json(metadata: dict[str, Any] | None) -> str:
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")
        try:
            return json.dumps(metadata, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("metadata must contain only JSON-compatible values") from error
