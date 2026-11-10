"""Reference only: the script that generated the stylized training images with FLUX.1 Kontext [dev]. It reads and writes S3 paths and
assumes specific GPUs, so adapt it before running; train.py downloads the finished dataset."""
#%%
import json
import numpy as np
with open("style_accepted.json", "r") as f:
    style_accepted_dict = json.load(f)
style_file = "selected_photostyles.csv"#"Complete_100_Photography_Styles__Detailed__updated.csv"
import pandas as pd
style_names = pd.read_csv(style_file)["Style Name"].tolist()
style_descriptions = pd.read_csv(style_file)["Description"].tolist()
style_acceptance_scores = [1-style_accepted_dict[name] for name in style_names]
style_acceptance_scores = [i if i > 0.5 else 0 for i in style_acceptance_scores]
# style_importance_weights = np.array(style_acceptance_scores) / np.sum(style_acceptance_scores)
#%%
import os
#pop distributed training flags from os.environ
os.environ.pop("RANK", None)
os.environ.pop("WORLD_SIZE", None)
os.environ.pop("MASTER_ADDR", None)
os.environ.pop("MASTER_PORT", None)
#%%
import torch
import random
from PIL import Image
from tqdm import tqdm
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import os
import multiprocessing as mp
from functools import partial
import io
from transformers import pipeline
import pandas as pd
from transformers import CLIPTokenizer
import boto3
from urllib.parse import urlparse
s3 = boto3.client("s3")

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def clip_token_count(text):
    return len(tokenizer(text)["input_ids"])
    #use t5 
#%%
img_home = "unsplash25k_512/"
out_home = "flux_style_ds/"
#%%
def get_existing_styles(image_name):
    #get which styles have not been used on this image
    path = f"{out_home}{image_name}/"
    #get all filenames in the path from s3
    pr = urlparse(path)
    bucket, key = pr.netloc, pr.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Malformed S3 URI: {path}")
    #check if the path exists
    try:
        files = s3.list_objects_v2(Bucket=bucket, Prefix=key)
        return [f["Key"].split("/")[-1].split(".")[0] for f in files["Contents"]]
    except Exception as e:
        return []
#%%
negative_prompt = """painting, drawing, illustration, sketch, cartoon, anime, 3D render, CGI, digital art, abstract, surreal, unrealistic transformation, morphed, warped 

monochrome, grayscale, desaturated, washed out, dull, lifeless, flat colors, uniform hue, solid color, single color overlay, monotone, sepia, duotone, color banding, posterization, oversaturated, neon, garish, unnatural colors 

text, watermark, signature, logo, border, frame, UI, blurry, pixelated, jpeg artifacts, over-sharpened, noise, plastic skin, fake look, glitch art"""
def infer(input_image, pipe, style_weights,seed=1526108804, guidance_scale=1.5, steps=28, bsize=10): #2.5
    """
    Perform image editing using the FLUX.1 Kontext pipeline.
    
    This function takes an input image and a text prompt to generate a modified version
    of the image based on the provided instructions. It uses the FLUX.1 Kontext model
    for contextual image editing tasks.
    
    Args:
        input_image (PIL.Image.Image): The input image to be edited. Will be converted
            to RGB format if not already in that format.
        prompt (str): Text description of the desired edit to apply to the image.
            Examples: "Remove glasses", "Add a hat", "Change background to beach".
        seed (int, optional): Random seed for reproducible generation. Defaults to 42.
            Must be between 0 and MAX_SEED (2^31 - 1).
        randomize_seed (bool, optional): If True, generates a random seed instead of
            using the provided seed value. Defaults to False.
        guidance_scale (float, optional): Controls how closely the model follows the
            prompt. Higher values mean stronger adherence to the prompt but may reduce
            image quality. Range: 1.0-10.0. Defaults to 2.5.
        steps (int, optional): Controls how many steps to run the diffusion model for.
            Range: 1-30. Defaults to 28.
    
    Returns:
        tuple: A 3-tuple containing:
            - PIL.Image.Image: The generated/edited image
            - int: The seed value used for generation (useful when randomize_seed=True)
            - gr.update: Gradio update object to make the reuse button visible
    
    Example:
        >>> edited_image, used_seed, button_update = infer(
        ...     input_image=my_image,
        ...     prompt="Add sunglasses",
        ...     seed=123,
        ...     randomize_seed=False,
        ...     guidance_scale=2.5
        ... )
    """
    # if randomize_seed:
    #     seed = random.randint(0, MAX_SEED)
    
    input_image = input_image.convert("RGB")
    #sample bsize style inds from style_weights no replacement
    if bsize >= np.count_nonzero(style_weights):
        inds = np.arange(len(style_weights))
    else:
        inds = np.random.choice(len(style_weights), size=bsize, p=style_weights, replace=False)
    style_batch = [styles[i] for i in inds]
    clip_ready_style_batch = [clip_ready_styles[i] for i in inds]
    #batch in groups of 10
    out = pipe(
        image=[input_image for _ in style_batch], 
        prompt=clip_ready_style_batch, 
        prompt_2=style_batch,
        negative_prompt=[negative_prompt_clip_ready for _ in style_batch],
        negative_prompt_2=[negative_prompt for _ in style_batch],
        guidance_scale=guidance_scale,
        width = input_image.size[0],
        height = input_image.size[1],
        num_inference_steps=steps,
        generator=[torch.Generator().manual_seed(seed) for _ in style_batch],
        max_area=input_image.size[0]*input_image.size[1],
        _auto_resize=False,
        # progress_bar_config={"disable": True},
    ).images
    return out, seed, inds
#%%
def prompt(name: str, description: str) -> str:
    return (f"Edit photofinishing and photostylization style to '{name}' : {description} ")
            # "Keep the original content, objects, and composition unchanged; only adjust the photo's colors, tones, grain, and texture to reflect this style.")
#%%
styles = [prompt(name, description) for name, description in zip(style_names, style_descriptions)]
CLIP_MAX = 77
summarizer = pipeline("summarization", model="t5-large", device_map="auto", torch_dtype=torch.bfloat16, batch_size=100)
def clip_token_count(text):
    return len(tokenizer(text)["input_ids"])
from tqdm import trange
def batch_summarize(texts, max_clip_tokens=CLIP_MAX, retry=3):
    out = texts
    for _ in trange(retry):
        todo_inds = [i for i, t in enumerate(out) if clip_token_count(t) > max_clip_tokens]
        if len(todo_inds) == 0:
            break
        todo_texts = [texts[i] for i in todo_inds]
        todo_summaries = summarizer(todo_texts,
                         min_length=40,          # T5 tokens
                         max_length=max_clip_tokens,
                         max_new_tokens=None,
                         do_sample=False)
        for i, summary in zip(todo_inds, todo_summaries):
            out[i] = summary['summary_text']
    return out
clip_ready_styles = batch_summarize(styles)
negative_prompt_clip_ready = batch_summarize([negative_prompt])[0]
assert all(clip_token_count(s) <= CLIP_MAX for s in clip_ready_styles)
del summarizer
#%%
import copy
pipe_org = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
def process_image_group(inp, out_home, img_home):
    (image_group, pipe_id) = inp
    """Process a group of images with a specific pipeline instance"""
    # Create pipeline instance for this worker
    pipe = copy.deepcopy(pipe_org).to(f"cuda:{allowed_devices[pipe_id]}")
    
    results = []
    for img_name in tqdm(image_group):
        try:
            # Load and preprocess image
            img_path = os.path.join(img_home, img_name)
            #find how many styles are left for this image
            existing_styles = get_existing_styles(img_name[:-4])
            is_style_todo = np.array([1 if style not in existing_styles else 0 for style in style_names])
            style_weights = is_style_todo * style_importance_weights
            if np.sum(style_weights) == 0:
                results.append(f"No styles left for {img_name}")
                continue
            style_weights = style_weights / np.sum(style_weights)
            img = Image.open(img_path)
            # img.thumbnail((512, 512), Image.Resampling.LANCZOS)
            #assert max dimension is 512
            assert max(img.size) <= 512, f"Image {img_name} has max dimension {max(img.size)}"
            # Run inference
            images_out, seed, inds = infer(img, pipe, style_weights)
                
            # Save original image to s3
            # img.save(f"{out_home}/{img_name}_0.jpg")

            img_name_no_ext = img_name.split(".")[0]
            out_path_dir = f"{out_home}{img_name_no_ext}/"   
            
            # Save generated images
            for j in range(len(images_out)):
                style_name = style_names[inds[j]]
                # save directly to s3 without saving to local
                img_buffer = io.BytesIO()
                images_out[j].save(img_buffer, format="JPEG", quality=99)
                images_out[j].save(f"{style_path}{img_name_no_ext}/{style_name}.jpg", format="JPEG", quality=99)
                img_buffer.seek(0)
                path_to_save = f"{out_path_dir}{style_name}.jpg"
                pr = urlparse(path_to_save)
                bucket, key = pr.netloc, pr.path.lstrip("/")
                s3.upload_fileobj(img_buffer, bucket, key)
                img_buffer.close()
            
            results.append(f"Processed {img_name}")
            
        except Exception as e:
            results.append(f"Error processing {img_name}: {str(e)}")
    
    # Clean up pipeline
    del pipe
    torch.cuda.empty_cache()
    
    return results

#%%
from collections import defaultdict
from score_cache_io import load_score_cache

img_path = "unsplash25k_512/"
style_path = "flux_style_ds/"
score_cache_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "score_cache.csv.gz",
)
scores = load_score_cache(score_cache_path)
#for each image, find out scores
scores_by_im = defaultdict(list)
scores_by_style = defaultdict(lambda: [0.5])
for (im_name, style_name), score in scores.items():
    if score > 0.5:
        scores_by_im[im_name].append((style_name, score))
        scores_by_style[style_name].append(score)
style_importance_weights = [np.median(scores_by_style[style_name])-0.5 for style_name in style_names]
style_importance_weights = np.array(style_importance_weights).clip(0)
style_importance_weights = style_importance_weights / np.sum(style_importance_weights)
#%%
# self.bad_im_names = [im_name for im_name in self.scores_by_im if len(self.scores_by_im[im_name]) < 5]
# files = sorted([os.path.join(img_path, f) for f in os.listdir(img_path) if f.endswith('.jpg') or f.endswith('.png')])
# scores_per_file = []
# sort
# files = [f for f in files if len(scores_by_im[f.split("/")[-1].split(".")[0]]) > 5]
#%%
# Create 8 pipeline instances and divide images into 8 groups
allowed_devices = [0, 1, 4, 5, 6, 7]
num_pipelines = len(allowed_devices)
imgs = [f for f in os.listdir(img_home) if f.endswith(".jpg")]
score_per_img = [len(scores_by_im[f.split("/")[-1].split(".")[0]]) for f in imgs]
#now sort by score_per_img
imgs = [f for _, f in sorted(zip(score_per_img, imgs), key=lambda x: x[0])]
#%%
image_groups = [imgs[i::num_pipelines] for i in range(num_pipelines)]

# Process images in parallel using multiprocessing
if __name__ == "__main__":
    # Set the start method to 'spawn' for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
    # Use multiprocessing to run inference in parallel
    with mp.Pool(processes=num_pipelines) as pool:
        # Create partial function with fixed arguments
        process_func = partial(process_image_group, out_home=out_home, img_home=img_home)
        
        # Map the work to the pool
        results = pool.map(process_func, [(group, i) for i, group in enumerate(image_groups)])
    
    # Print results
    for i, result in enumerate(results):
        print(f"Pipeline {i} results:")
        for msg in result:
            print(f"  {msg}")
        print()

#%%
