import asyncio
import logging
import traceback
from abc import ABC, abstractmethod
from asyncio import Queue, Semaphore
from copy import deepcopy
from typing import Tuple, Callable, Coroutine

from FlagEmbedding import BGEM3FlagModel, FlagAutoModel

logger = logging.getLogger(__name__)

class EmbeddingModel(ABC):
    @abstractmethod
    def text_sparse_embed(self, batch_sentences: list[str]) -> list[dict[int, float]]:
        ...


class AsyncEmbeddingEngine:
    sparse_queue: Queue = Queue(maxsize=32768)
    dense_queue: Queue = Queue(maxsize=32768)
    batch_size: int
    max_token_length: int
    max_workers: Semaphore
    def __init__(self, model_name_or_path: str, batch_size: int, max_token_length: int = 8192, max_workers_num: int = 8):
        # 通过model path 加载model
        # self.model: BGEM3FlagModel = BGEM3FlagModel(model_name_or_path, use_fp16=True, query_max_length=max_token_length)
        self.model = FlagAutoModel.from_finetuned(model_name_or_path=model_name_or_path, use_fp16=True, query_max_length=max_token_length)
        self.batch_size = batch_size
        self.max_token_length = max_token_length
        self._run = False
        self.max_workers = Semaphore(max_workers_num)
        logger.debug("max workers = %s", max_workers_num)

    async def _execute_sparse_batch(self, tasks: list[tuple]):
        task_sentences: [str] = []
        task_callbacks: [Callable] = []
        for task in tasks:
            task_sentences.append(task[0])
            task_callbacks.append(task[1])
        try:
            # tokens_list = [self.model.tokenizer.encode(
            #     s, max_length=self.max_length, return_tensors="pt"
            # ) for s in task_sentences]
            # tokens_list: list[list[int]] = self.model.tokenizer.batch_encode_plus(
            #     max_length=self.max_length,
            #     truncation=TruncationStrategy.LONGEST_FIRST,
            #     batch_text_or_text_pairs=deepcopy(task_sentences)).get(
            #     "input_ids")
            async with self.max_workers:
                tokens_list: list[list[int]] = [[0] for _ in range(len(task_sentences))]
                logger.debug("tokens_list = %s", tokens_list)
                _sparse_embeddings: list[dict] = self.model.encode(
                    max_length=self.max_token_length,
                    queries=task_sentences,
                    batch_size=self.batch_size,
                    return_dense=False,
                    return_sparse=True,
                    return_colbert_vecs=False
                ).get("lexical_weights")
                logger.debug("_sparse_embeddings = %s", _sparse_embeddings)
        except Exception as e:
            traceback.print_exc()
            logger.error("Error in _execute_sparse_batch: %s", e.__traceback__)
            for _callback in task_callbacks:
                _callback(None, [], [], e)
            return

        for sentence, task_callback, sparse_embedding, tokens in zip(task_sentences, task_callbacks, _sparse_embeddings,
                                                                     tokens_list):
            task_callback(sentence, sparse_embedding, tokens)

    async def _execute_dense_batch(self, tasks: list[tuple]):
        task_sentences: [str] = []
        task_callbacks: [Callable] = []
        for task in tasks:
            task_sentences.append(task[0])
            task_callbacks.append(task[1])

        try:
            # tokens_list = [self.model.tokenizer.encode(
            #     s, max_length=self.max_length, truncation=TruncationStrategy.LONGEST_FIRST, return_tensors="pt"
            # ) for s in task_sentences]
            # tokens_list: list[list[int]] = self.model.tokenizer.batch_encode_plus(
            #     truncation=TruncationStrategy.LONGEST_FIRST,
            #     max_length=self.max_length,
            #     batch_text_or_text_pairs=deepcopy(task_sentences)).get(
            #     "input_ids")
            async with self.max_workers:
                tokens_list: list[list[int]] = [[0] for _ in range(len(task_sentences))]

                _dense_embeddings: list[list[float]] = self.model.encode(
                    max_length=self.max_token_length,
                    queries=task_sentences,
                    batch_size=self.batch_size,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False).get("dense_vecs").tolist()
        except Exception as e:
            traceback.print_exc()
            logger.error("Error in _execute_dense_batch: %s", e)
            for _callback in task_callbacks:
                _callback(None, [], [], e)
            return

        for sentence, task_callback, dense_embedding, tokens in zip(task_sentences, task_callbacks, _dense_embeddings,
                                                                    tokens_list):
            task_callback(sentence, dense_embedding, tokens)

    async def _consume_task(self, batch_size: int, embedding_queue: Queue, execution: Callable[...,Coroutine]):
        batch_tasks: list[tuple] = []
        while self._run:
            logger.debug("batch_tasks = %s", batch_tasks)
            tasks_length = len(batch_tasks)
            if tasks_length <= 0:  # batch_tasks 为空.
                logger.debug("batch_task is empty, wait get task from queue")
                task = await embedding_queue.get()  # 阻塞一直等待有任务.
                if task is not None:
                    sentence, callback = task
                    batch_tasks.append((sentence, callback))
                    embedding_queue.task_done()
                else:
                    embedding_queue.task_done()
                    break
            elif tasks_length >= batch_size:  # batch_tasks 达到批处理size
                # execution(batch_tasks)
                logger.debug("save documents")
                embedding_task = asyncio.create_task(execution(deepcopy(batch_tasks)))
                batch_tasks.clear()
            else:
                logger.debug("batch_task %s, queue size %s", len(batch_tasks), embedding_queue.qsize())
                is_empty = embedding_queue.empty()
                if is_empty:  #
                    # execution(batch_tasks)
                    logger.debug("save documents")
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

    async def stop(self):
        self._run = False
        await self.sparse_queue.put(None)
        await self.dense_queue.put(None)

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
