import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, SiglipModel, SiglipVisionModel, AutoImageProcessor, AutoModel, AutoTokenizer, AutoConfig

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

model_path = "/workspace/models/siglip-so400m-patch14-384"

config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_path)
print(config)

model = SiglipModel.from_pretrained(pretrained_model_name_or_path=model_path, torch_dtype=torch.float32, device_map="cuda", attn_implementation="sdpa")
model_1 = SiglipVisionModel.from_pretrained(pretrained_model_name_or_path=model_path, torch_dtype=torch.float32, device_map="cuda", attn_implementation="sdpa")
processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=model_path)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
image_1 = Image.open("/workspace/models/test_images/ldh_1.jpeg", mode='r').convert("RGB")
image_2 = Image.open("/workspace/models/test_images/cat_4.jpeg", mode='r').convert("RGB")

input_1 = processor(images=image_1, return_tensors="pt").to(device)
input_2 = processor(images=image_2, return_tensors="pt").to(device)

embedding_1 = model.get_image_features(**input_1)
e_1_1 = model_1(**input_1,return_dict=True, output_hidden_states=True)[1]


embedding_1 = torch.nn.functional.normalize(embedding_1, p=2, dim=-1)
e_1_1= torch.nn.functional.normalize(e_1_1, p=2, dim=-1)
print(embedding_1)
print(e_1_1)

embedding_2 = model.get_image_features(**input_2)
embedding_2 = torch.nn.functional.normalize(embedding_2, p=2, dim=-1)

score = np.dot(embedding_1.cpu().detach().numpy(), embedding_2.cpu().detach().numpy().T)
print(score)

