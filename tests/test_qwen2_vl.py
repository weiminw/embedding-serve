import torch
from PIL import Image
from transformers import AutoProcessor, SiglipModel, AutoImageProcessor, AutoModel, AutoTokenizer, Qwen2VLModel, \
    Qwen2VLForConditionalGeneration, Qwen2VLProcessor
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

model = Qwen2VLModel.from_pretrained("/workspace/models/Qwen2-VL-7B-Instruct-AWQ").to(device)
processor = Qwen2VLProcessor.from_pretrained("/workspace/models/Qwen2-VL-7B-Instruct-AWQ")
tokenizer = AutoTokenizer.from_pretrained("/workspace/models/Qwen2-VL-7B-Instruct-AWQ")
image = Image.open("/workspace/models/refine2.jpg", mode='r')
print(processor)
# width = 512
# ratio = (width / float(image.size[0]))
# height = int((float(image.size[1]) * float(ratio)))
# img = image.resize((width, height), Image.Resampling.LANCZOS)
with torch.no_grad():
    inputs = processor(images=image,return_tensors="pt").to(device)
    image_features = model.encode_image(**inputs)
    print(image_features.size())
    x = F.normalize(image_features, p=2, dim=1)
    # print(image_features)
    print(len(x.detach().cpu().numpy()[0]))