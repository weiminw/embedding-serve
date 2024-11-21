import numpy as np
import torch
from PIL import Image
from transformers import ViTModel, ViTImageProcessor, ViTFeatureExtractor, ViTForImageClassification, ViTConfig, \
    ViTPreTrainedModel, ViTHybridModel, ViTHybridImageProcessor, AutoConfig
from transformers.models.auto.image_processing_auto import image_processors

def norm(x: torch.Tensor):
    return torch.nn.functional.normalize(x, p=2, dim=-1)

model_path = "/workspace/models/vit-huge-patch14-224-in21k"
config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_path)
print(config)
image_processor = ViTImageProcessor.from_pretrained(pretrained_model_name_or_path=model_path, return_tensors="pt",device_map="cuda")
model = ViTModel.from_pretrained(pretrained_model_name_or_path=model_path, attn_implementation="sdpa", torch_dtype=torch.float32, device_map="cuda")

image_1 = Image.open("/workspace/models/test_images/ldh_1.jpeg").convert("RGB")
input_1 = image_processor(images=image_1, return_tensors="pt").to("cuda")

image_2 = Image.open("/workspace/models/test_images/ldh_2.jpeg").convert("RGB")
input_2 = image_processor(images=image_2, return_tensors="pt").to("cuda")
print(type(input_2))

output_1 = model(**input_1, return_dict=True)
output_2 = model(**input_2, return_dict=True)

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