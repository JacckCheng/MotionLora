from diffusers import StableDiffusionPipeline
from peft import LoraConfig
import torch
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
import os

# 可以尝试写到config文件中
model_path = "/home/chengtianle/MotionLora/stable-diffusion-v1-5"
lora_weights_path = "/home/chengtianle/MotionLora/configs/checkpoints"
output_path = "/home/chengtianle/MotionLora/outputs"
prompt = "peaple, sit"

# 从本地路径加载模型
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# 执行推理
output = pipe(prompt).images[0]

# 保存生成的图像
output.save(os.path.join(output_path, 'sit0.png'))