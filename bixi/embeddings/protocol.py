from typing import Union, Literal

from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[list[str], str]
    embedding_type: Literal['text_dense', 'text_sparse', 'image', 'video'] = 'text_dense'


class DenseEmbeddingData(BaseModel):
    object: Literal['embedding'] = 'embedding'
    index: int = 0
    embedding: list[float]

class SparseEmbeddingData(BaseModel):
    object: Literal['embedding'] = 'embedding'
    index: int = 0
    embedding: dict[str,float]

class EmbeddingUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0

class EmbeddingResponse(BaseModel):
    object: Literal['list'] = 'list'
    model: str = ""
    data: list[Union[DenseEmbeddingData,SparseEmbeddingData]]
    usage: EmbeddingUsage


