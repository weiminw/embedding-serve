from collections import defaultdict
from typing import Union

import numpy
import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, is_torch_npu_available, AutoModel, AutoModelForTokenClassification, \
    BatchEncoding, SiglipModel, ViTModel
from transformers.modeling_outputs import TokenClassifierOutput
from typing_extensions import Literal


class EmbeddingModel(torch.nn.Module):
    _support_embedding_types: list[Literal["text_dense_embedding", "text_sparse_embedding", "image_embedding"]] = []
    def __init__(self, model_name_or_path: str, use_fp16: bool, max_token_length: int, device: str = None, trust_remote_code: bool = True):
        super(EmbeddingModel, self).__init__()
        if device:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif is_torch_npu_available():
                self.device = torch.device("npu")
            else:
                self.device = torch.device("cpu")
                use_fp16 = False
        self.max_token_length = max_token_length
        self.config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        model_type = self.config.model_type
        if model_type == "new":  # Alibaba-NLP: GTE
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16 if use_fp16 else None,
                device_map=self.device)
            self._support_embedding_types = ["text_dense_embedding", "text_sparse_embedding"]
        elif model_type == "siglip":
            self.model = SiglipModel.from_pretrained(
                pretrained_model_name_or_path=model_name_or_path,
                torch_dtype=torch.float16 if use_fp16 else None,
                device_map=self.device,
                attn_implementation="sdpa")
            self._support_embedding_types = ["image_embedding"]
        elif model_type == "vit":
            self.model = ViTModel.from_pretrained(
                pretrained_model_name_or_path=model_name_or_path,
                attn_implementation="sdpa",
                torch_dtype=torch.float16 if use_fp16 else None,
                device_map=self.device)
            self._support_embedding_types = ["image_embedding"]
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        print(self.config)
        print(self.config.model_type)

    @staticmethod
    def last_token_pool(last_hidden_states: torch.Tensor,
                        attention_masks: torch.Tensor) -> torch.Tensor:
        """Last token pooling method.

        Args:
            last_hidden_states (torch.Tensor): The last hidden state of the model.
            attention_masks (torch.Tensor): Attention mask. Defaults to :data:`None`.

        Returns:
            torch.Tensor: The embedding vectors after pooling.
        """
        left_padding = (attention_masks[:, -1].sum() == attention_masks.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_masks.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def _process_token_weights(self, token_weights: np.ndarray, input_ids: list):
        # convert to dict
        result = defaultdict(int)
        unused_tokens = {self.tokenizer.cls_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id,
                         self.tokenizer.unk_token_id}
        # token_weights = np.ceil(token_weights * 100)
        for w, idx in zip(token_weights, input_ids):
            if idx not in unused_tokens and w > 0:
                token = self.tokenizer.decode([int(idx)])
                if w > result[token]:
                    result[token] = w
        return dict(result)

    def _dense_embedding(self, model_output: TokenClassifierOutput, inputs: dict[str, torch.Tensor]) -> numpy.array:
        last_hidden_states = model_output.hidden_states[-1]
        attention_masks = inputs["attention_mask"]
        embeddings = self.last_token_pool(last_hidden_states=last_hidden_states, attention_masks=attention_masks)
        embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings_norm.detach().cpu().numpy()

    def _sparse_embedding(self, model_output: TokenClassifierOutput, inputs: dict[str, torch.Tensor]) -> numpy.array:
        print("logits: %s",model_output.logits)
        token_weights = torch.relu(model_output.logits).squeeze(-1)
        print("token_weights: %s",token_weights)
        token_weights = torch.nn.functional.normalize(token_weights, p=2, dim=-1)
        print("token_weights norm: %s", token_weights)
        token_weights = list(map(self._process_token_weights, token_weights.detach().cpu().numpy().tolist(),
                                 inputs['input_ids'].detach().cpu().numpy().tolist()))
        return token_weights

    @staticmethod
    def _compute_sparse_scores(x: dict, y: dict):
        scores = 0
        for token, weight in x.items():
            if token in y:
                scores += weight * y[token]
        return scores

    def compute_sparse_scores(self, x: list[dict], y: list[dict]) -> list[float]:
        scores = [self._compute_sparse_scores(embedding_1, embedding_2) for embedding_1, embedding_2 in zip(x,y)]
        return scores

    def compute_dense_scores(self, x: numpy.array, y: numpy.array) -> numpy.array:
        scores = numpy.dot(x, y.T)

    @torch.no_grad()
    def encode(self, texts: list[str], max_length:int = 8192, batch_size: int = 16, embedding_type: Literal['text_dense_embedding','text_sparse_embedding'] = 'text_dense_embedding') -> Union[list[list[float]], list[dict]]:
        num_texts = len(texts)
        for n, i in enumerate(range(0, num_texts, batch_size)):
            chunk_texts: list[str] = texts[i: i + batch_size]
            chunk_encoding: BatchEncoding = self.tokenizer(chunk_texts, padding=True, truncation=True, return_tensors="pt",
                                    max_length=max_length)  # {'input_ids': tensor([[14990,  1879]]), 'attention_mask': tensor([[1, 1]])}
            chunk_inputs:dict[str, torch.Tensor] = {k: v.to(self.device) for k, v in chunk_encoding.items()}
            model_output = self.model(**chunk_inputs, return_dict=True, output_hidden_states=True)
            if embedding_type == 'text_dense_embedding':
                chunk_dense = self._dense_embedding(model_output, chunk_inputs)
                return chunk_dense
            else:
                chunk_sparse = self._sparse_embedding(model_output, chunk_inputs)
                return chunk_sparse

    @property
    def support_types(self):
        return self._support_embedding_types


