import asyncio
import logging
import time
import traceback
from abc import ABC, abstractmethod
from asyncio import Queue, Semaphore
from copy import deepcopy
from io import BytesIO

from encodings.base64_codec import base64_decode
from typing import Tuple, Callable, Coroutine

from PIL import Image
from transformers import AutoModel

from bixi.embeddings.models.base import EmbeddingModel

logger = logging.getLogger(__name__)

class AsyncEmbeddingEngine:
    sparse_queue: Queue = Queue(maxsize=32768)
    dense_queue: Queue = Queue(maxsize=32768)
    image_queue: Queue = Queue(maxsize=32768)
    batch_size: int
    dense_embedding_max_token_length: int
    sparse_embedding_max_token_length: int
    max_dense_workers: Semaphore
    max_sparse_workers: Semaphore
    def __init__(self, model: EmbeddingModel, batch_size: int, dense_embedding_max_token_length: int = 8192, sparse_embedding_max_token_length: int = 512, max_workers_num: int = 8):
        # 通过model path 加载model
        # self.model: BGEM3FlagModel = BGEM3FlagModel(model_name_or_path, use_fp16=True, query_max_length=max_token_length)
        # self.model = FlagAutoModel.from_finetuned(model_name_or_path=model_name_or_path, use_fp16=True, query_max_length=dense_embedding_max_token_length)
        # self.model: EmbeddingModel = EmbeddingModel(model_name_or_path, use_fp16=True, max_token_length=dense_embedding_max_token_length)
        logger.debug("loaded model: %s", model)
        self.model = model
        self.batch_size = batch_size
        self.dense_embedding_max_token_length = dense_embedding_max_token_length
        self.sparse_embedding_max_token_length = sparse_embedding_max_token_length
        self._run = False
        self.max_dense_workers = Semaphore(int(max_workers_num/2))
        self.max_sparse_workers = Semaphore(int(max_workers_num/2))
        logger.debug("max workers = %s", max_workers_num)

    async def _execute_sparse_batch(self, tasks: list[tuple]):
        task_sentences: [str] = []
        task_callbacks: [Callable] = []
        for task in tasks:
            task_sentences.append(task[0])
            task_callbacks.append(task[1])

        _start_time = time.time()
        try:
            async with self.max_sparse_workers:
                tokens_list: list[list[int]] = [[0] for _ in range(len(task_sentences))]
                _sparse_embeddings: list[dict] = self.model.encode_text(
                    max_length=self.sparse_embedding_max_token_length,
                    texts=task_sentences,
                    batch_size=self.batch_size,
                    encode_type='sparse'
                )
                for sentence, task_callback, sparse_embedding, tokens in zip(task_sentences, task_callbacks,
                                                                             _sparse_embeddings,
                                                                             tokens_list):
                    task_callback(sentence, sparse_embedding, tokens)
        except Exception as e:
            logger.exception("Error in _execute_sparse_batch: %s", e)
            for _callback in task_callbacks:
                _callback(None, [], [], e)
            return
        _end_time = time.time()
        logger.debug("finished sparse embedding %s sentences, used %s 毫秒", len(task_sentences), (_end_time - _start_time) * 1000)

    async def _execute_dense_batch(self, tasks: list[tuple]):
        task_sentences: [str] = []
        task_callbacks: [Callable] = []
        for task in tasks:
            task_sentences.append(task[0])
            task_callbacks.append(task[1])
        _start_time = time.time()
        try:
            async with self.max_dense_workers:
                tokens_list: list[list[int]] = [[0] for _ in range(len(task_sentences))]

                _dense_embeddings: list[list[float]] = self.model.encode_text(
                    max_length=self.dense_embedding_max_token_length,
                    texts=task_sentences,
                    batch_size=self.batch_size,
                    encode_type='dense'
                )
                for sentence, task_callback, dense_embedding, tokens in zip(task_sentences, task_callbacks,
                                                                            _dense_embeddings, tokens_list):
                    task_callback(sentence, dense_embedding, tokens)
        except Exception as e:
            logger.exception("Error in _execute_dense_batch: %s", e)
            for _callback in task_callbacks:
                _callback(None, [], [], e)
            return
        _end_time = time.time()
        logger.debug("finished dense embedding %s sentences, used %s 毫秒", len(task_sentences), (_end_time - _start_time) * 1000)

    async def _execute_image_batch(self, tasks: list[tuple]):
        task_images: [Image] = []
        task_callbacks: [Callable] = []
        for task in tasks:
            task_images.append(task[0])
            task_callbacks.append(task[1])
        _start_time = time.time()
        try:
            async with self.max_dense_workers:
                tokens_list: list[list[int]] = [[0] for _ in range(len(task_images))]

                _dense_embeddings: list[list[float]] = self.model.encode_image(
                    max_length=self.dense_embedding_max_token_length,
                    images=task_images,
                    batch_size=self.batch_size,
                )
                for image, task_callback, dense_embedding, tokens in zip(task_images, task_callbacks,
                                                                            _dense_embeddings, tokens_list):
                    task_callback(image, dense_embedding, tokens)
        except Exception as e:
            logger.exception("Error in _execute_dense_batch: %s", e)
            for _callback in task_callbacks:
                _callback(None, [], [], e)
            return
        _end_time = time.time()
        logger.debug("finished dense embedding %s sentences, used %s 毫秒", len(task_images), (_end_time - _start_time) * 1000)

    async def _consume_task(self, batch_size: int, embedding_queue: Queue, execution: Callable[...,Coroutine]):
        batch_tasks: list[tuple] = []
        while self._run:
            tasks_length = len(batch_tasks)
            if tasks_length <= 0:  # batch_tasks 为空.
                task = await embedding_queue.get()  # 阻塞一直等待有任务.
                if task is not None:
                    sentence, callback = task
                    batch_tasks.append((sentence, callback))
                    embedding_queue.task_done()
                else:
                    embedding_queue.task_done()
                    break
            elif tasks_length >= batch_size:  # batch_tasks 达到批处理size
                embedding_task = asyncio.create_task(execution(deepcopy(batch_tasks)))
                batch_tasks.clear()
            else:
                is_empty = embedding_queue.empty()
                if is_empty:  #
                    embedding_task = asyncio.create_task(execution(deepcopy(batch_tasks)))
                    batch_tasks.clear()
                else:
                    task = await embedding_queue.get()  # 阻塞一直等待有任务.
                    if task is not None:
                        sentence, callback = task
                        batch_tasks.append((sentence, callback))
                        embedding_queue.task_done()
                    else:
                        embedding_queue.task_done()
                        break

    async def start(self):
        self._run = True
        run_sparse_task = asyncio.create_task(self._consume_task(self.batch_size, self.sparse_queue, self._execute_sparse_batch))
        run_dense_task = asyncio.create_task(self._consume_task(self.batch_size, self.dense_queue, self._execute_dense_batch))
        run_image_task = asyncio.create_task(self._consume_task(self.batch_size, self.image_queue, self._execute_image_batch))

    async def stop(self):
        self._run = False
        await self.sparse_queue.put(None)
        await self.dense_queue.put(None)
        await self.image_queue.put(None)

    async def text_sparse_embed(self, sentences: list[str]) -> Tuple[list[dict[int, float]], int]:
        embedding_results: list[tuple] = []
        # 定义信号量, 确保该批处理完成再返回
        semaphore = asyncio.Semaphore(0)
        task_error: dict = {}

        # 定义回调函数
        def callback(_sentence: str, _embedding: dict[int, float], _tokens: list[int], _error: Exception = None):
            if _error is not None:
                semaphore.release()
                task_error["error"] = _error
            embedding_results.append((_embedding, _tokens))
            if len(embedding_results) == len(sentences):
                semaphore.release()

        # 提交
        for sentence in sentences:
            await self.sparse_queue.put(item=(sentence, callback))

        await semaphore.acquire()
        if "error" in task_error:
            raise task_error["error"]
        embeddings: list[dict[int, float]] = []
        tokens_num: int = 0
        for embedding_result_values in embedding_results:
            embedding, tokens = embedding_result_values
            embeddings.append(embedding)
            tokens_num += (len(tokens) - 2)
        return embeddings, tokens_num

    async def text_dense_embed(self, sentences: list[str]) -> Tuple[list[list[float]], int]:
        embedding_results: list[tuple] = []
        # 定义信号量, 确保该批处理完成再返回
        semaphore = asyncio.Semaphore(0)
        task_error: dict = {}

        # 定义回调函数
        def callback(_sentence: str, _embedding: list[list[float]], _tokens: list[int], _error: Exception = None):
            if _error is not None:
                semaphore.release()
                task_error["error"] = _error
            embedding_results.append((_embedding, _tokens))
            if len(embedding_results) == len(sentences):
                semaphore.release()

        # 提交
        for sentence in sentences:
            await self.dense_queue.put(item=(sentence, callback))

        await semaphore.acquire()
        if "error" in task_error:
            raise task_error["error"]
        embeddings: list[list[float]] = []
        tokens_num: int = 0
        for embedding_result in embedding_results:
            embedding, tokens = embedding_result
            embeddings.append(embedding)
            tokens_num += (len(tokens) - 2)
        return embeddings, tokens_num

    async def image_dense_embed(self, images: list[Image]) -> Tuple[list[list[float]], int]:
        embedding_results: list[tuple] = []
        # 定义信号量, 确保该批处理完成再返回
        semaphore = asyncio.Semaphore(0)
        task_error: dict = {}

        # 定义回调函数
        def callback(_sentence: Image, _embedding: list[list[float]], _tokens: list[int], _error: Exception = None):
            if _error is not None:
                semaphore.release()
                task_error["error"] = _error
            embedding_results.append((_embedding, _tokens))
            if len(embedding_results) == len(images):
                semaphore.release()

        # 提交
        for image in images:
            await self.image_queue.put(item=(image, callback))

        await semaphore.acquire()
        if "error" in task_error:
            raise task_error["error"]
        embeddings: list[list[float]] = []
        tokens_num: int = 0
        for embedding_result in embedding_results:
            embedding, tokens = embedding_result
            embeddings.append(embedding)
            tokens_num += (len(tokens) - 2)
        return embeddings, tokens_num
