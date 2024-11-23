import base64
import logging.config
import re
from argparse import Namespace
from contextlib import asynccontextmanager
from io import BytesIO

from PIL import Image
from typing import Any

import uvicorn
from fastapi import FastAPI
from starlette.datastructures import State
from starlette.requests import Request
from starlette.responses import JSONResponse

from bixi.embeddings import SparseEmbeddingData, \
    DenseEmbeddingData,  EmbeddingUsage
from bixi.embeddings.engines import AsyncEmbeddingEngine
from bixi.embeddings.models import EmbeddingModel
from bixi.embeddings import EmbeddingRequest, EmbeddingResponse
from bixi.embeddings.settings import configure_logging, get_logging_configuration

logger = logging.getLogger("bixi.embeddings.serve")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Waiting for embedding server startup")
        embedding_engine = app.state.engine
        await embedding_engine.start()
        logger.info("Embedding server started")
        yield

    finally:
        embedding_engine = app.state.engine
        await embedding_engine.stop()
        logger.info("Embedding server shutdown")


def init_app_state(state: State, args: Namespace) -> None:
    model_name_or_path = args.model
    batch_size = args.batch_size
    max_token_length = args.max_token_len
    max_workers_num = args.max_workers_num
    log_level = args.log_level.upper()
    configure_logging(logger_name="bixi", level=log_level)
    configure_logging(logger_name="uvicorn", level="INFO")
    configure_logging(logger_name="uvicorn.error", level="INFO")
    configure_logging(logger_name="uvicorn.access", level="INFO")
    logger.info("args: %s", args)
    model = EmbeddingModel(model_name_or_path, use_fp16=True, max_token_length=max_token_length)
    embedding_engine = AsyncEmbeddingEngine(model=model, batch_size=batch_size, max_workers_num=max_workers_num)
    state.engine = embedding_engine

embedding_app = FastAPI(lifespan=lifespan)

class RequestError(Exception):
    def __init__(self, message: str, code: int = 400, data: Any = None):
        self.message = message
        self.code = code
        self.data = data
        super().__init__(message)

def base64_to_image(base64_str: str) -> Image:
    _s = base64_str.split(",")
    _head = _s[0]
    if _head is not None and re.match(r"^data:image\/(" + "|".join(["jpeg", "jpg", "png", "gif", "bmp", "tiff", "tif", "webp", "svg"]) + r");base64", _head):
        _content = _s[1]
        return Image.open(BytesIO(base64.b64decode(_content)))
    else:
        logger.error("Input string should be valid base64 encoded string and  start with 'data:image/xxx;base64', currently supported image types are: jpeg, jpg, png, gif, bmp, tiff, tif, webp, svg")
        raise RequestError(f"Input string should be valid base64 encoded string and  start with 'data:image/xxx;base64', currently supported image types are: jpeg, jpg, png, gif, bmp, tiff, tif, webp, svg")

@embedding_app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})

@embedding_app.exception_handler(RequestError)
async def exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=400, content={"error": str(exc)})
@embedding_app.post("/v1/embeddings")
async def execute_embedding(request: EmbeddingRequest, raw_request: Request):
    logger.debug("received embedding request")
    if isinstance(request.input, list):
        inputs = request.input
    else:
        inputs = [request.input]
    embedding_type = request.embedding_type
    embedding_engine: AsyncEmbeddingEngine = raw_request.app.state.engine
    if embedding_type not in embedding_engine.model.support_embedding_types:
        logger.error("embedding type %s is not supported by the model", embedding_type)
        raise RequestError(f"embedding type {embedding_type} is not supported by the model")

    if embedding_type == 'text_dense_embedding':
        embeddings: list[list[float]]
        usage: int
        embeddings, usage = await embedding_engine.text_dense_embed(sentences=inputs)
        dense_embedding_datas = [DenseEmbeddingData(embedding = item, index = i) for i, item in enumerate(embeddings)]

        response = EmbeddingResponse(data = dense_embedding_datas, model=request.model, usage=EmbeddingUsage(prompt_tokens=usage, total_tokens=usage))
        logger.debug("dense embeddings: %s", len(inputs))
        return response
    elif embedding_type == 'text_sparse_embedding':
        embeddings: list[dict[str, float]]
        usage: int
        embeddings, usage = await embedding_engine.text_sparse_embed(sentences=inputs)
        sparse_embedding_datas = [SparseEmbeddingData(embedding=item, index=i) for i, item in enumerate(embeddings)]

        response = EmbeddingResponse(data=sparse_embedding_datas, model=request.model,
                                           usage=EmbeddingUsage(prompt_tokens=usage, total_tokens=usage))
        logger.debug("sparse embeddings: %s", len(inputs))
        return response
    elif embedding_type == 'image_embedding':
        embeddings: list[list[float]]
        usage: int
        images:[Image] = []
        for _input in inputs:
            image = base64_to_image(_input)
            images.append(image)

        embeddings, usage = await embedding_engine.image_dense_embed(images=images)
        dense_embedding_datas = [DenseEmbeddingData(embedding=item, index=i) for i, item in enumerate(embeddings)]

        response = EmbeddingResponse(data=dense_embedding_datas, model=request.model,
                                          usage=EmbeddingUsage(prompt_tokens=usage, total_tokens=usage))
        logger.debug("image embeddings: %s", len(inputs))
        return response
    else:
        logger.error("embedding type %s is not implemented", embedding_type)
        raise RequestError(f"embedding type {embedding_type} is not implemented")
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embedding Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", type=int, default=8000, help="端口号")
    parser.add_argument("--model", type=str, help="huggingface模型ID或者本地模型的路径")
    parser.add_argument("--max-token-len", type=int, default=8192, help="最大embedding token长度")
    parser.add_argument("--batch-size", type=int, default=256, help="批处理大小")
    parser.add_argument("--max-workers-num", type=int, default=8, help="并发工作协程数")
    parser.add_argument("--served-model-name", type=str, help="服务模型名称")
    parser.add_argument("--api-ssl-key", type=str, default=None, help="API SSL密钥文件路径")
    parser.add_argument("--log-level", type=str, default="DEBUG", help="日志级别")
    args = parser.parse_args()
    init_app_state(embedding_app.state, args)

    uvicorn.run(embedding_app, host=args.host, port=args.port, log_config=get_logging_configuration())