import torch
from PIL import Image
from transformers import AutoProcessor, SiglipModel, AutoImageProcessor, AutoModel, AutoTokenizer, Qwen2VLModel, \
    Qwen2VLForConditionalGeneration, Qwen2VLProcessor, AutoConfig, Qwen2VLConfig
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

config = AutoConfig.from_pretrained("/workspace/models/Qwen2-VL-7B-Instruct-AWQ")
config.max_position_embeddings = 2048
print(config)
model = Qwen2VLForConditionalGeneration.from_pretrained("/workspace/models/Qwen2-VL-7B-Instruct-AWQ", config=config)
model.to(device)
print(model.eval())
processor = Qwen2VLProcessor.from_pretrained("/workspace/models/Qwen2-VL-7B-Instruct-AWQ")
tokenizer = AutoTokenizer.from_pretrained("/workspace/models/Qwen2-VL-7B-Instruct-AWQ")
image = Image.open("/workspace/models/refine2.jpg")

conversation = [
    {
        "role":"user",
        "content":[
            {
                "type":"image",
            },
            {
                "type":"text",
                "text":"Describe this image."
            }
        ]
    }
]
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(text=[text_prompt], images=[image], padding=False, return_tensors="pt")
inputs.to(device)
output_ids = model.generate(**inputs, max_new_tokens=128)

    # image_features = model.encode_image(**inputs)
    # print(image_features.size())
    # x = F.normalize(image_features, p=2, dim=1)
    # # print(image_features)
    # print(len(x.detach().cpu().numpy()[0]))