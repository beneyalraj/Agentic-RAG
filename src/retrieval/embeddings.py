import logging
from typing import List, Union, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from config.config import settings

logger = logging.getLogger(__name__)
class EmbeddingModel:
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL, device: Optional[str] = None, default_batch_size: int = 32):

        self.model_name = model_name
        self.default_batch_size = default_batch_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        self.device = device
        logger.info(f"Loading model '{model_name}' on device: {device}")

        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def encode_single(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)

    def encode_batch(self, texts: List[str], batch_size: Optional[int] = None, show_progress_bar: bool = False) -> np.ndarray:

        batch_size = batch_size or self.default_batch_size
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=show_progress_bar,)

_embedding_model = None

def get_embedding_model(model_name: str = settings.EMBEDDING_MODEL,device: Optional[str] = None) -> EmbeddingModel:

    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel(model_name, device)
    return _embedding_model