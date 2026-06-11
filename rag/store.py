"""DocumentStore: embeddings generation and vector indexing with ChromaDB.

Embeddings are computed via the OpenAI embeddings API and passed to
Chroma explicitly (we do NOT use Chroma's default embedding function,
which would try to download a local model at runtime — bad for a
lightweight hosted deployment).
"""

from openai import OpenAI

import chromadb

EMBEDDING_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 100  # chunks per embeddings API request


class DocumentStore:
    """Owns the vector index: add chunks, run similarity search.

    Uses an in-memory (ephemeral) Chroma collection. For a corpus of a
    few hundred chunks this is fast, free, and avoids persistence
    headaches on hosting platforms with ephemeral file systems.
    """

    def __init__(self, api_key: str):
        """
        Args:
            api_key: OpenAI API key used for the embeddings endpoint.
        """
        self._client = OpenAI(api_key=api_key, timeout=30)
        self._chroma = chromadb.EphemeralClient()
        self._collection = self._chroma.get_or_create_collection(
            name="emergency_protocol",
            metadata={"hnsw:space": "cosine"},
        )
        # source name -> number of chunks indexed from it
        self._sources: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, batching requests to the API."""
        vectors: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            response = self._client.embeddings.create(
                model=EMBEDDING_MODEL, input=batch
            )
            vectors.extend(item.embedding for item in response.data)
        return vectors

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[dict]) -> int:
        """Embed and index a list of chunks.

        Args:
            chunks: dicts with "id", "text", and "source" keys (as
                produced by DocumentIngestor.chunk).

        Returns:
            The number of chunks added.
        """
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embeddings = self._embed(texts)

        self._collection.add(
            ids=[c["id"] for c in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"source": c["source"]} for c in chunks],
        )

        for c in chunks:
            self._sources[c["source"]] = self._sources.get(c["source"], 0) + 1
        return len(chunks)

    def remove_source(self, source: str) -> None:
        """Remove all chunks belonging to one document."""
        self._collection.delete(where={"source": source})
        self._sources.pop(source, None)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 4) -> list[dict]:
        """Find the top-k chunks most similar to the query.

        Args:
            query: the user's natural-language question.
            k: number of chunks to return.

        Returns:
            A list of dicts: {"text", "source", "score"} ordered from
            most to least relevant. Score is cosine similarity (0–1).
        """
        if self.chunk_count == 0:
            return []

        query_vector = self._embed([query])[0]
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(k, self.chunk_count),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for text, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append(
                {
                    "text": text,
                    "source": meta["source"],
                    # Chroma returns cosine *distance*; convert to similarity.
                    "score": round(1 - distance, 3),
                }
            )
        return hits

    # ------------------------------------------------------------------
    # Introspection (used by the UI)
    # ------------------------------------------------------------------

    @property
    def sources(self) -> dict[str, int]:
        """Mapping of indexed document names to their chunk counts."""
        return dict(self._sources)

    @property
    def chunk_count(self) -> int:
        """Total number of chunks currently indexed."""
        return sum(self._sources.values())
