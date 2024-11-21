import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, XLMRobertaModel
from FlagEmbedding import FlagAutoModel

def norm(x: torch.Tensor):
    return torch.nn.functional.normalize(x, p=2, dim=-1)

def score(x: torch.Tensor, y: torch.Tensor):
    return np.dot(x.detach().cpu().numpy(), y.detach().cpu().numpy().T)

def score_2(x, y):
    return np.dot(x,y.T)

device="cuda"
model_path = "/workspace/models/Baai/bge-m3"

model = XLMRobertaModel.from_pretrained(pretrained_model_name_or_path=model_path, torch_dtype=torch.float16, device_map=device)
flag_model = FlagAutoModel.from_finetuned(model_name_or_path=model_path, use_fp16=True, device=device)
config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_path)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, device_map=device)

string_1 = "hello world"
string_2 = "hi"

input_1 = tokenizer(text=string_1, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
input_2 = tokenizer(text=string_2, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

embeddings_1 = model(**input_1, return_dict=True)
print(embeddings_1)
embeddings_2 = model(**input_2, return_dict=True)

print("string 1: %s", norm(embeddings_1["last_hidden_state"][:,0]))
print("string 2: %s", norm(embeddings_2["last_hidden_state"][:,0]))

print("\n---------------------\n")

print("string 1 pool: %s", norm(embeddings_1["pooler_output"]))
print("string 2 pool: %s", norm(embeddings_2["pooler_output"]))

print("\n---------------------\n")

print("score 1: %s", score(norm(embeddings_1["last_hidden_state"][:,0]), norm(embeddings_2["last_hidden_state"][:,0])))
print("score 2: %s", score(norm(embeddings_1["pooler_output"]), norm(embeddings_2["pooler_output"])))

embeddings_1_feature = flag_model.encode(queries=[string_1])["dense_vecs"]
embeddings_2_feature = flag_model.encode(queries=[string_2])["dense_vecs"]

print("score 3: %s", score_2(embeddings_1_feature, embeddings_2_feature))



