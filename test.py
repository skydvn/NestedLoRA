from huggingface_hub import hf_hub_download
import timm

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_IPV6"] = "1"

model = timm.create_model(
    "deit_tiny_patch16_224",
    pretrained=True
)
