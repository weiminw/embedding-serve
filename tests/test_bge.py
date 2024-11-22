from collections import defaultdict

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
sparse_linear_model = torch.nn.Linear(model.config.hidden_size, 1, dtype=torch.float16).to(device)
sparse_linear_model.load_state_dict(torch.load(f"{model_path}/sparse_linear.pt"))
flag_model = FlagAutoModel.from_finetuned(model_name_or_path=model_path, use_fp16=True, device=device,  return_sparse_embedding=True)
config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_path)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, device_map=device)

string_1 = "This is big world, China is a big country."
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

embeddings_1_sparse = flag_model.encode(queries=[string_1], return_sparse=True)["lexical_weights"]
print("embeddings 1 sparse: ", embeddings_1_sparse)
for k,v in dict(embeddings_1_sparse[0]).items():
    print("k={}, v={}".format(tokenizer.decode([int(k)]),v))
print("score 3: %s", score_2(embeddings_1_feature, embeddings_2_feature))

print("\n------- sparse -------\n")
model_output = embeddings_1
token_weights = torch.relu(sparse_linear_model(model_output.last_hidden_state))
# print("token weights = {}, token weight shape = {}".format(token_weights.squeeze(-1),token_weights.squeeze(-1).shape))
# print("input_ids: ", input_1["input_ids"].squeeze(-1).shape)
# print("input ids size 0: ", input_1["input_ids"].size(0))
# print("input ids size 1: ", input_1["input_ids"].size(1))
# print("vocab size: ", model.config.vocab_size)
s = torch.zeros(input_1["input_ids"].size(0), input_1["input_ids"].size(1), model.config.vocab_size, dtype=torch.float16, device=device)
print("s size: ", s.shape)
# # print("input ids unsqueeze: ", input_1["input_ids"].unsqueeze(-1).shape)
s = torch.scatter(input=s,dim=-1,index=input_1["input_ids"].unsqueeze(-1),src=token_weights)
print("scatter: ", s[0,0,0], s[0,1,3293])
unused_tokens = [
            tokenizer.cls_token_id, tokenizer.eos_token_id,
            tokenizer.pad_token_id, tokenizer.unk_token_id
        ]
print("unused tokens: ", unused_tokens)
sparse_embedding = torch.max(s, dim=-1).values.squeeze(-1)
print("sparse embedding: ", sparse_embedding)
sparse_embedding[:, unused_tokens] *= 0.0
print("sparse embedding: ", sparse_embedding.squeeze(-1).shape)
# print("token_weights: ", token_weights)
result = defaultdict(int)
unused_tokens = set([tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id,
                         tokenizer.unk_token_id])
for w,idx in zip(sparse_embedding.squeeze(-1).detach().cpu().numpy().tolist()[0], input_1["input_ids"].squeeze(-1).detach().cpu().numpy().tolist()[0]):
    print("w={}, idx={}, token={}".format(w, idx,tokenizer.decode([int(idx)])))

    if idx not in unused_tokens and w > 0:
        token = tokenizer.decode([int(idx)])
        if w > result[token]:
            result[token] = w
print("result: ", dict(result))




