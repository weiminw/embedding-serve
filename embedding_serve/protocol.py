from typing import Union

from pydantic import BaseModel

class SparseEmbeddingRequest(BaseModel):
    model: str
    input: Union[list[str], str]

class SparseEmbeddingResponse(BaseModel):
    embeddings: list[dict[int, float]]