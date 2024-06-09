from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np

def setup_qdrant_collection(collection_name):
    client = QdrantClient(host="localhost", port=6333)
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=100, distance=models.Distance.COSINE)
    )
    return client, collection_name

def recommend_items(client, collection_name, query_vector, positive_ids, negative_ids, limit):
    recommendations = client.recommend(
        collection_name=collection_name,
        query_vector=query_vector,
        positive=positive_ids,
        negative=negative_ids,
        limit=limit
    )
    return recommendations
