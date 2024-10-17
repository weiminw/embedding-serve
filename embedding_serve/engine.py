import os
import queue
from abc import ABC, abstractmethod
from queue import Queue

from FlagEmbedding import FlagModel, BGEM3FlagModel

from embedding_serve.protocol import SparseEmbeddingResponse


class EmbeddingModel(ABC):
    @abstractmethod
    def text_sparse_embed(self, batch_sentences: list[str]) -> list[dict[int, float]]:
        ...




class AsyncEmbeddingEngine:
    def __init__(self, model_name_or_path: str, batch_size: int):
        # 通过model path 加载model
        self.model = BGEM3FlagModel(model_name_or_path, use_fp16=True)
        self.batch_size = batch_size

    async def text_sparse_embed(self, sentences: list[str]) -> list[dict[int, float]]:
        return self.model.encode(sentences, batch_size=self.batch_size, return_dense=False, return_sparse=True, return_colbert_vecs=False).get("lexical_weights")

    async def text_dense_embed(self, sentences: list[str]):
        raise NotImplementedError("Text dense embedding is not implemented yet.")