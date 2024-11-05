import asyncio
import queue
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from select import error
from typing import Tuple, Callable

import torch.cuda
from FlagEmbedding import BGEM3FlagModel
from transformers.tokenization_utils_base import TruncationStrategy

from bixi.embeddings.logging_config import logger


class EmbeddingModel(ABC):
    @abstractmethod
    def text_sparse_embed(self, batch_sentences: list[str]) -> list[dict[int, float]]:
        ...


class AsyncEmbeddingEngine:
    sparse_queue: Queue = Queue(maxsize=32768)
    dense_queue: Queue = Queue(maxsize=32768)
    embed_executor = ThreadPoolExecutor(max_workers=8)

    def __init__(self, model_name_or_path: str, batch_size: int):
        # 通过model path 加载model
        self.model = BGEM3FlagModel(model_name_or_path, use_fp16=True)
        self.batch_size = batch_size
        self._run = False

    def _execute_sparse_batch(self, tasks: list[tuple]):
        task_sentences: [str] = []
        task_callbacks: [Callable] = []
        for task in tasks:
            task_sentences.append(task[0])
            task_callbacks.append(task[1])
        try:
            tokens_list: list[list[int]] = self.model.tokenizer.batch_encode_plus(
                max_length=1024,
                truncation=TruncationStrategy.LONGEST_FIRST,
                batch_text_or_text_pairs=task_sentences).get(
                "input_ids")
            logger.debug("tokens_list = %s", tokens_list)
            _sparse_embeddings: list[dict] = self.model.encode(
                task_sentences,
                batch_size=self.batch_size,
                return_dense=False,
                return_sparse=True,
                return_colbert_vecs=False).get("lexical_weights")
            logger.debug("_sparse_embeddings = %s", _sparse_embeddings)
        except Exception as e:
            logger.error("Error in _execute_sparse_batch: %s", e)
            for _callback in task_callbacks:
                _callback(None, [], [], e)
            return

        for sentence, task_callback, sparse_embedding, tokens in zip(task_sentences, task_callbacks, _sparse_embeddings,
                                                                     tokens_list):
            task_callback(sentence, sparse_embedding, tokens)

    def _execute_dense_batch(self, tasks: list[tuple]):
        task_sentences: [str] = []
        task_callbacks: [Callable] = []
        for task in tasks:
            task_sentences.append(task[0])
            task_callbacks.append(task[1])

        try:
            tokens_list: list[list[int]] = self.model.tokenizer.batch_encode_plus(
                truncation=TruncationStrategy.LONGEST_FIRST,
                max_length=8192,
                batch_text_or_text_pairs=task_sentences).get(
                "input_ids")

            _dense_embeddings: list[list[float]] = self.model.encode(
                task_sentences,
                batch_size=self.batch_size,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False).get("dense_vecs")
        except Exception as e:
            logger.error("Error in _execute_dense_batch: %s", e)
            for _callback in task_callbacks:
                _callback(None, [], [], e)
            return

        for sentence, task_callback, dense_embedding, tokens in zip(task_sentences, task_callbacks, _dense_embeddings,
                                                                    tokens_list):
            task_callback(sentence, dense_embedding, tokens)

    def _consume_task(self, batch_size: int, embedding_queue: Queue, execution: Callable):
        batch_tasks: list[tuple] = []
        while self._run:
            tasks_length = len(batch_tasks)
            if tasks_length <= 0:  # batch_tasks 为空.
                task = embedding_queue.get()  # 阻塞一直等待有任务.
                if task is not None:
                    sentence, callback = task
                    batch_tasks.append((sentence, callback))
            elif tasks_length >= batch_size:  # batch_tasks 达到批处理size
                execution(batch_tasks)
                batch_tasks.clear()
            else:
                if embedding_queue.empty():  #
                    execution(batch_tasks)
                    batch_tasks.clear()
                else:
                    task = embedding_queue.get()  # 阻塞一直等待有任务.
                    if task is not None:
                        sentence, callback = task
                        batch_tasks.append((sentence, callback))

    async def start(self):
        self._run = True
        loop = asyncio.get_event_loop()
        loop.run_in_executor(self.embed_executor, self._consume_task, self.batch_size, self.sparse_queue,
                             self._execute_sparse_batch)
        loop.run_in_executor(self.embed_executor, self._consume_task, self.batch_size, self.dense_queue,
                             self._execute_dense_batch)

    async def stop(self):
        self._run = False
        self.sparse_queue.put(None)
        self.dense_queue.put(None)

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
            self.sparse_queue.put(item=(sentence, callback), block=False)

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
            self.dense_queue.put(item=(sentence, callback), block=False)

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
