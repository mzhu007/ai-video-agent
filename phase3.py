import json
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).parent
INPUT_FILE = PROJECT_DIR / "output" / "cinematic_prompts.json"
OUTPUT_DIR = PROJECT_DIR / "output" / "images"

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

DEVICE = "mps"
WIDTH = 768
HEIGHT = 432
NUM_INFERENCE_STEPS = 25
GUIDANCE_SCALE = 7.5
SEED = 42


# --------------------------------------------------
# Load scene prompts
# --------------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    video_data = json.load(f)

scenes = video_data["scenes"]

if not scenes:
    raise ValueError("No scenes found in cinematic_prompts.json")

scene = scenes[0]

positive_prompt = scene["positive_prompt"]
negative_prompt = scene["negative_prompt"]

scene_number = scene["scene_number"]

print(f"Generating image for Scene {scene_number}...")


# --------------------------------------------------
# Load Stable Diffusion
# --------------------------------------------------

print(f"Loading model: {MODEL_ID}")

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True,
)

pipe.enable_attention_slicing()
pipe = pipe.to(DEVICE)


# --------------------------------------------------
# Generate image
# --------------------------------------------------

generator = torch.Generator(device="cpu").manual_seed(SEED)

print("Running Stable Diffusion inference...")

image = pipe(
    prompt=positive_prompt,
    negative_prompt=negative_prompt,
    width=WIDTH,
    height=HEIGHT,
    num_inference_steps=NUM_INFERENCE_STEPS,
    guidance_scale=GUIDANCE_SCALE,
    generator=generator,
).images[0]


# --------------------------------------------------
# Save image
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_file = OUTPUT_DIR / f"scene_{scene_number:02d}.png"

image.save(output_file)

print(f"Image saved: {output_file}")