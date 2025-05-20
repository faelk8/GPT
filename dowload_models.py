import urllib.request
import os

models = [  # "gpt2-small-124M.pth",
    # "gpt2-medium-355M.pth",
    # "gpt2-large-774M.pth",
    "gpt2-xl-1558M.pth",
    # "gpt2-small-124M.safetensors",
    # "gpt2-medium-355M.safetensors",
    # "gpt2-large-774M.safetensors",
    "gpt2-xl-1558M.safetensors",]

for m in models:
    url = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{m}"

    dest_path = f"src/models/{m}"

    if not os.path.exists(dest_path):
        urllib.request.urlretrieve(url, dest_path)
        print(f"Downloaded to {dest_path}")
