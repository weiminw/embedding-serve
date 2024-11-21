import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, SiglipModel, SiglipVisionModel, AutoImageProcessor, AutoModel, AutoTokenizer, AutoConfig

def norm(x: torch.Tensor):
    return torch.nn.functional.normalize(x, p=2, dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

model_path = "/workspace/models/siglip-so400m-patch14-384"

config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_path)
print(config)

model_all = SiglipModel.from_pretrained(pretrained_model_name_or_path=model_path, torch_dtype=torch.float32, device_map="cuda", attn_implementation="sdpa")
model = SiglipVisionModel.from_pretrained(pretrained_model_name_or_path=model_path, torch_dtype=torch.float32, device_map="cuda", attn_implementation="sdpa")
processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=model_path)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
image_1 = Image.open("/workspace/models/test_images/ldh_1.jpeg", mode='r').convert("RGB")
image_2 = Image.open("/workspace/models/test_images/cat_3.jpeg", mode='r').convert("RGB")

input_1 = processor(images=image_1, return_tensors="pt").to(device)
input_2 = processor(images=image_2, return_tensors="pt").to(device)

output_1 = model(**input_1, return_dict=True)
output_2 = model(**input_2, return_dict=True)

output_1_feature = model_all.get_image_features(**input_1)
print("feature 1: %s", output_1_feature)
output_2_feature = model_all.get_image_features(**input_2)
print("feature 2: %s", output_2_feature)


# print(output_1)
embedding_1 = output_1["last_hidden_state"][:,0]
embedding_1_pool = output_1["pooler_output"]
print("embedding 1: %s", embedding_1)
print("embedding 1 pool: %s", embedding_1_pool)
print("embedding 1 norm: %s", torch.nn.functional.normalize(embedding_1, p=2, dim=-1))
print("embedding 1 pool norm: %s", torch.nn.functional.normalize(embedding_1_pool, p=2, dim=-1))

print("\n-----------------------\n")
embedding_2 = output_2["last_hidden_state"][:,0]
embedding_2_pool = output_2["pooler_output"]
print("embedding 2: %s", embedding_2)
print("embedding 2 pool: %s", embedding_2_pool)
print("embedding 2 norm: %s", torch.nn.functional.normalize(embedding_2, p=2, dim=-1))
print("embedding 2 pool norm: %s", torch.nn.functional.normalize(embedding_2_pool, p=2, dim=-1))



score = np.dot(norm(embedding_1).cpu().detach().numpy(), norm(embedding_2).cpu().detach().numpy().T)
print("score: %s",score)
score = np.dot(norm(embedding_1_pool).cpu().detach().numpy(), norm(embedding_2_pool).cpu().detach().numpy().T)
print("score: %s",score)

score = np.dot(norm(embedding_1).cpu().detach().numpy(), norm(embedding_2).cpu().detach().numpy().T)
print(score)
score = np.dot(norm(embedding_1_pool).cpu().detach().numpy(), norm(embedding_2_pool).cpu().detach().numpy().T)
print(score)
score = np.dot(norm(output_1_feature).detach().cpu().numpy(), norm(output_2_feature).detach().cpu().numpy().T)
print(score)

