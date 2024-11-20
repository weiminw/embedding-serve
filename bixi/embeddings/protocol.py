from typing import Union, Literal

from pydantic import BaseModel


class DenseEmbeddingRequest(BaseModel):
    model: str
    input: Union[list[str], str]


class DenseEmbeddingData(BaseModel):
    object: Literal['list'] = 'embedding'
    index: int = 0
    embedding: list[float]

class EmbeddingUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0

class DenseEmbeddingResponse(BaseModel):
    object: Literal['dict'] = 'list'
    model: str = ""
    data: list[DenseEmbeddingData]
    usage: EmbeddingUsage


class SparseEmbeddingRequest(BaseModel):
    model: str
    input: Union[list[str], str]


class SparseEmbeddingData(BaseModel):
    object: Literal['list'] = 'embedding'
    index: int = 0
    embedding: dict[str,float]


class SparseEmbeddingResponse(BaseModel):
    object: Literal['dict'] = 'list'
    model: str = ""
    data: list[SparseEmbeddingData]
    usage: EmbeddingUsage
