import numpy as np
import torch
from PIL import Image
from transformers import ViTModel, ViTImageProcessor, ViTFeatureExtractor, ViTForImageClassification, ViTConfig, \
    ViTPreTrainedModel, ViTHybridModel, ViTHybridImageProcessor, AutoConfig
from transformers.models.auto.image_processing_auto import image_processors

model_path = "/workspace/models/vit-huge-patch14-224-in21k"
config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_path)
print(config)
image_processor = ViTImageProcessor.from_pretrained(pretrained_model_name_or_path=model_path, return_tensors="pt",device_map="cuda")
model = ViTModel.from_pretrained(pretrained_model_name_or_path=model_path, attn_implementation="sdpa", torch_dtype=torch.float32, device_map="cuda")

image_1 = Image.open("/workspace/models/test_images/cat_4.jpeg").convert("RGB")
input_1 = image_processor(images=image_1, return_tensors="pt").to("cuda")

image_2 = Image.open("/workspace/models/test_images/ldh_3.jpeg").convert("RGB")
input_2 = image_processor(images=image_2, return_tensors="pt").to("cuda")

output_1 = model(**input_1)
output_2 = model(**input_2)

embedding_1 = output_1[1]
embedding_1 = torch.nn.functional.normalize(embedding_1, p=2, dim=-1)
print(embedding_1.shape)

embedding_2 = output_2[1]
embedding_2 = torch.nn.functional.normalize(embedding_2, p=2, dim=-1)
print(embedding_2.shape)



si = np.dot(embedding_1.cpu().detach().numpy(), embedding_2.cpu().detach().numpy().T)
print(si)