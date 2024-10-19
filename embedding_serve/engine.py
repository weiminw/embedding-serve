
from abc import ABC, abstractmethod
from typing import Tuple

from FlagEmbedding import FlagModel, BGEM3FlagModel




class EmbeddingModel(ABC):
    @abstractmethod
    def text_sparse_embed(self, batch_sentences: list[str]) -> list[dict[int, float]]:
        ...




class AsyncEmbeddingEngine:
    def __init__(self, model_name_or_path: str, batch_size: int):
        # 通过model path 加载model
        self.model = BGEM3FlagModel(model_name_or_path, use_fp16=True)
        self.batch_size = batch_size

    async def text_sparse_embed(self, sentences: list[str]) -> Tuple[list[dict[int, float]], int]:
        tokens_list: list[list[int]] = self.model.tokenizer.batch_encode_plus(batch_text_or_text_pairs=sentences).get(
            "input_ids")
        tokens_nums: list[int] = [len(tokens) for tokens in tokens_list]
        _sparse_embeddings: list[dict] = self.model.encode(sentences, batch_size=self.batch_size, return_dense=False, return_sparse=True, return_colbert_vecs=False).get("lexical_weights")
        return [{int(key): value for key, value in item.items()} for item in _sparse_embeddings], sum(tokens_nums)


    async def text_dense_embed(self, sentences: list[str]) ->Tuple[list[list[float]], int]:
        
        tokens_list: list[list[int]] = self.model.tokenizer.batch_encode_plus(batch_text_or_text_pairs=sentences).get("input_ids")
        tokens_nums: list[int] = [ len(tokens) for tokens in tokens_list]
        _dense_embeddings: list[list[float]] = self.model.encode(sentences, batch_size=self.batch_size, return_dense=True, return_sparse=False, return_colbert_vecs=False).get("dense_vecs")
        return _dense_embeddings, sum(tokens_nums)