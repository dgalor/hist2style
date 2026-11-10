#%%
import os
import numpy as np
from PIL import Image
from collections import defaultdict
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.models import vgg19, VGG19_Weights
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from score_cache_io import load_score_cache, save_score_cache
target_home = "flux_style_ds/"
reference_home = "unsplash25k_512/"
im_names = [f.split(".")[0] for f in os.listdir(reference_home)]
#%%
vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
vgg.eval().cuda()
preprocessing = VGG19_Weights.IMAGENET1K_V1.transforms()
#%%
score_cache_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "score_cache.csv.gz",
)
if os.path.exists(score_cache_path):
    score_cache = load_score_cache(score_cache_path)
else:
    score_cache = {}
#%%
im_style_pairs = []
for im_name in tqdm(im_names):
    im_path = os.path.join(target_home, im_name)
    if os.path.exists(im_path):
        for style_name in os.listdir(im_path):
            pair = (im_name, style_name.split(".")[0])
            if pair not in score_cache:
                im_style_pairs.append(pair)
print(f"Found {len(im_style_pairs)} image-style pairs left to process")
#%%
print(len([i for i in score_cache.values() if i > 0.5]), "images with score > 0.5")
print(len(score_cache), "total image-style pairs")
#%%
def get_score(im_name, style_names, cache):
    im = Image.open(os.path.join(reference_home, im_name + ".jpg"))
    styles = [Image.open(os.path.join(target_home, im_name, style_name + ".jpg")) for style_name in style_names]
    images = [im] + styles
    images = torch.stack([preprocessing(img) for img in images], dim=0)
    with torch.no_grad():
        feats = vgg(images.cuda()).cpu().flatten(1)
    cos_sim = F.cosine_similarity(feats[0], feats[1:], dim=1).tolist()
    for style_name, score in zip(style_names, cos_sim):
        cache[(im_name, style_name)] = score

#group pairs by im_name
grouped_pairs = defaultdict(list)
for pair in im_style_pairs:
    grouped_pairs[pair[0]].append(pair[1])
#process in parallel
with ThreadPoolExecutor(max_workers=16) as executor:
    future_to_pair = {executor.submit(get_score, im_name, style_names, score_cache): im_name for im_name, style_names in grouped_pairs.items()}
    for future in tqdm(as_completed(future_to_pair), total=len(grouped_pairs), desc="Processing image-style pairs"):
        future.result()
#save cache
save_score_cache(score_cache_path, score_cache)
#%%
print(len([i for i in score_cache.values() if i > 0.5]), "images with score > 0.5")
print(len(score_cache), "total image-style pairs")
#%%
#prev total ds is 0.6 yield, current is 0.52 yield.