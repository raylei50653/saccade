"""Tests for the Chroma vector store client (saccade.storage.chroma_store)."""

# scope: storage
# function: behavior
# lifecycle: active

from __future__ import annotations

from unittest.mock import MagicMock

from saccade.storage.chroma_store import ChromaStore


def _store_with_collection(collection: MagicMock) -> ChromaStore:
    store = ChromaStore.__new__(ChromaStore)
    store.collection = collection
    return store


def test_add_memory_passes_one_embedding_per_document() -> None:
    collection = MagicMock()
    store = _store_with_collection(collection)
    embedding = [0.1, 0.2, 0.3]

    store.add_memory("person entering", {}, doc_id="m1", embedding=embedding)

    collection.add.assert_called_once_with(
        documents=["person entering"],
        metadatas=[collection.add.call_args.kwargs["metadatas"][0]],
        ids=["m1"],
        embeddings=[embedding],
    )


def test_hybrid_query_includes_zero_start_time_filter() -> None:
    collection = MagicMock()
    collection.query.return_value = {"ids": [[]]}
    store = _store_with_collection(collection)

    store.hybrid_query(query_embedding=[0.1, 0.2], start_time=0.0)

    collection.query.assert_called_once_with(
        n_results=5,
        where={"timestamp": {"$gte": 0.0}},
        query_embeddings=[[0.1, 0.2]],
    )
