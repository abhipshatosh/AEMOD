import tempfile
import unittest
from pathlib import Path

from vector_db import VectorDatabase


class VectorDatabaseTests(unittest.TestCase):
    def test_upsert_and_search_by_cosine_and_euclidean(self):
        with VectorDatabase(":memory:") as database:
            database.upsert("docs", "a", [1, 0], {"title": "east"})
            database.upsert("docs", "b", [0, 1], {"title": "north"})
            database.upsert("docs", "c", [-1, 0])

            cosine = database.search("docs", [2, 0], limit=2)
            self.assertEqual([result.id for result in cosine], ["a", "b"])
            self.assertAlmostEqual(cosine[0].distance, 0)
            self.assertEqual(cosine[0].metadata, {"title": "east"})

            euclidean = database.search("docs", [1, 0], metric="euclidean", limit=1)
            self.assertEqual([result.id for result in euclidean], ["a"])
            self.assertEqual(euclidean[0].distance, 0)

            database.upsert("docs", "a", [0, 2], {"title": "updated"})
            self.assertEqual(
                database.get("docs", "a").metadata, {"title": "updated"}
            )

    def test_data_persists_across_connections(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.sqlite3"
            with VectorDatabase(path) as database:
                database.upsert("docs", "one", [0.25, 0.75], {"page": 3})

            with VectorDatabase(path) as reopened:
                record = reopened.get("docs", "one")
                self.assertEqual(record.vector, (0.25, 0.75))
                self.assertEqual(record.metadata, {"page": 3})

    def test_invalid_dimension_does_not_change_existing_record(self):
        with VectorDatabase(":memory:") as database:
            database.upsert("docs", "one", [1, 2])
            with self.assertRaises(ValueError):
                database.upsert("docs", "one", [1, 2, 3])
            self.assertEqual(database.get("docs", "one").vector, (1.0, 2.0))

    def test_validation_and_delete(self):
        with VectorDatabase(":memory:") as database:
            with self.assertRaises(ValueError):
                database.upsert("docs", "one", [float("nan")])
            database.upsert("docs", "one", [1])
            with self.assertRaises(ValueError):
                database.search("docs", [1], limit=-1)
            with self.assertRaises(KeyError):
                database.search("missing", [1])
            self.assertTrue(database.delete("docs", "one"))
            self.assertFalse(database.delete("docs", "one"))
            self.assertIsNone(database.get("docs", "one"))


if __name__ == "__main__":
    unittest.main()
