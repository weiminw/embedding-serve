import asyncio
from contextlib import asynccontextmanager

from argparse import Namespace
from typing import Tuple

import uvicorn
from FlagEmbedding import FlagModel, BGEM3FlagModel
from fastapi import FastAPI
from starlette.datastructures import State
from starlette.requests import Request
from uvicorn.config import LOGGING_CONFIG

from embedding_serve.engine import AsyncEmbeddingEngine
from embedding_serve.protocol import SparseEmbeddingRequest, SparseEmbeddingResponse, SparseEmbeddingData, \
    DenseEmbeddingData, DenseEmbeddingResponse, EmbeddingUsage
from embedding_serve.logging_config import logger, LOG_FORMAT, LOG_COLORS

LOGGING_CONFIG["formatters"]["default"]["fmt"] = LOG_FORMAT
LOGGING_CONFIG["formatters"]["default"]["use_colors"] = True

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("embedding server start...")
    try:
        # embedding_engine = app.state.engine
        # await embedding_engine.start()
        yield

    finally:
        embedding_engine = app.state.engine
        await embedding_engine.stop()
        logger.info("embedding server shutdown")


def init_app_state(state: State, args: Namespace) -> None:
    model_name_or_path = args.model
    # batch_size = args.batch_size
    embedding_engine = AsyncEmbeddingEngine(model_name_or_path=model_name_or_path, batch_size=32)

    state.engine = embedding_engine
    asyncio.run(embedding_engine.start())


app = FastAPI(lifespan=lifespan)

@app.post("/v1/embeddings/sparse")
async def sparse_embeddings(request: SparseEmbeddingRequest, raw_request: Request) -> SparseEmbeddingResponse:
    if isinstance(request.input, list):
        inputs = request.input
    else:
        inputs = [request.input]
    embedding_engine: AsyncEmbeddingEngine = raw_request.app.state.engine
    embeddings: list[list[float]]
    usage: int
    embeddings, usage = await embedding_engine.text_sparse_embed(sentences=inputs)
    sparse_embedding_datas = [SparseEmbeddingData(embedding = item, index = i) for i, item in enumerate(embeddings)]

    response = SparseEmbeddingResponse(data = sparse_embedding_datas, model=request.model, usage=EmbeddingUsage(prompt_tokens=usage, total_tokens=usage))
    return response


@app.post("/v1/embeddings")
async def dense_embeddings(request: SparseEmbeddingRequest, raw_request: Request) -> DenseEmbeddingResponse:
    if isinstance(request.input, list):
        inputs = request.input
    else:
        inputs = [request.input]
    embedding_engine: AsyncEmbeddingEngine = raw_request.app.state.engine
    embeddings: list[list[float]]
    usage: int
    embeddings, usage = await embedding_engine.text_dense_embed(sentences=inputs)
    dense_embedding_datas = [DenseEmbeddingData(embedding = item, index = i) for i, item in enumerate(embeddings)]

    response = DenseEmbeddingResponse(data = dense_embedding_datas, model=request.model, usage=EmbeddingUsage(prompt_tokens=usage, total_tokens=usage))
    return response

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FastAPI应用程序")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", type=int, default=8000, help="端口号")
    parser.add_argument("--model", type=str, default="BAAI/bge-m3", help="huggingface模型ID或者本地模型的路径")
    parser.add_argument("--served_model_name", type=str, default="bge", help="服务模型名称")
    parser.add_argument("--api_ssl_key", type=str, default=None, help="API SSL密钥文件路径")
    args = parser.parse_args()
    init_app_state(app.state, args)
    uvicorn.run(app, host=args.host, port=args.port)