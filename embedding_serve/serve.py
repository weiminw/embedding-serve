from contextlib import asynccontextmanager

import colorama
import uvicorn
from fastapi import FastAPI
from uvicorn.config import LOGGING_CONFIG

from embedding_serve.protocol import SparseEmbeddingRequest
from logging_config import logger, LOG_FORMAT, LOG_COLORS

LOGGING_CONFIG["formatters"]["default"]["fmt"] = LOG_FORMAT
LOGGING_CONFIG["formatters"]["default"]["use_colors"] = True

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("embedding server start...")
    try:
        yield
    finally:
        logger.info("embedding server shutdown")


app = FastAPI(lifespan=lifespan)

@app.post("/v1/embeddings/sparse")
async def sparse_embeddings(request: SparseEmbeddingRequest) -> list[dict[int,float]]:
    logger.debug("fdfd")

    return []

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="FastAPI应用程序")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", type=int, default=8000, help="端口号")
    parser.add_argument("--model_name", type=str, default="example_model", help="模型名称")
    parser.add_argument("--served_model_name", type=str, default="example_served_model", help="服务模型名称")
    parser.add_argument("--api_ssl_key", type=str, default=None, help="API SSL密钥文件路径")

    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)