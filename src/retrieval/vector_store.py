import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pinecone
from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)

class PineconeVectorStore:
    def __init__(self, api_key: str, index_name: str, cloud: str = "aws", region: str = "us-west-2", dimension: int = 384, metric: str = "cosine",):

        self.client = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.cloud = cloud
        self.region = region
        self.dimension = dimension
        self.metric = metric
        self.index = None

    def create_index_if_not_exists(self) -> None:
        existing_indexes = [idx.name for idx in self.client.list_indexes()]

        if self.index_name not in existing_indexes:
            logger.info(
                f"Creating index '{self.index_name}' with dimension={self.dimension}, "
                f"metric='{self.metric}', cloud='{self.cloud}', region='{self.region}'"
            )
            self.client.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
        else:
            logger.info(f"Index '{self.index_name}' already exists.")

        self.index = self.client.Index(self.index_name)
        logger.info(f"Connected to index '{self.index_name}'")

    def upsert_vectors(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]], batch_size: int = 100, ) -> None:

        if self.index is None:
            raise RuntimeError("Index not initialised. Call create_index_if_not_exists() first.")

        total = len(vectors)
        for i in range(0, total, batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
            logger.debug(f"Upserted batch {i//batch_size + 1} ({len(batch)} vectors)")

    def query(self, query_embedding: List[float], top_k: int = 10, include_metadata: bool = True, filter: Optional[Dict] = None,) -> List[Dict[str, Any]]:

        if self.index is None:
            raise RuntimeError("Index not initialised. Call create_index_if_not_exists() first.")

        response = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=include_metadata, filter=filter,)
        return response["matches"]

    def describe_index_stats(self) -> Dict[str, Any]:

        if self.index is None:
            raise RuntimeError("Index not initialised.")
        return self.index.describe_index_stats()