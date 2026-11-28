
import json
from pathlib import Path

from ltx_video.inference import InferenceConfig, infer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

PROMPT_FILE = PROJECT_DIR / "output" / "cinematic_prompts.json"

IMAGE_FILE = PROJECT_DIR / "output" / "images" / "scene_01.png"

MODEL_CONFIG = (
    PROJECT_DIR
    / "third_party"
    / "LTX-Video"
    / "configs"
    / "ltxv-2b-0.9.8-distilled.yaml"
)

OUTPUT_DIR = PROJECT_DIR / "output" / "videos"

SEED = 42

# First generation: short, low-resolution test
WIDTH = 512
HEIGHT = 288
NUM_FRAMES = 49
FRAME_RATE = 24


# --------------------------------------------------
# Load Phase 2 cinematic prompt
# --------------------------------------------------

def load_prompt():
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenes = data["scenes"]

    scene = next(
        item for item in scenes
        if item["scene_number"] == 1
    )

    return (
        scene["positive_prompt"],
        scene.get("negative_prompt", ""),
    )


# --------------------------------------------------
# Generate video from image
# --------------------------------------------------

def main():

    print("=" * 55)
    print("PHASE 4: IMAGE-TO-VIDEO GENERATION")
    print("=" * 55)

    # Validate inputs
    for path in [PROMPT_FILE, IMAGE_FILE, MODEL_CONFIG]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    positive_prompt, negative_prompt = load_prompt()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Input image: {IMAGE_FILE}")
    print(f"Model config: {MODEL_CONFIG}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Resolution: {WIDTH} x {HEIGHT}")
    print(f"Duration: {NUM_FRAMES / FRAME_RATE:.2f} seconds")
    print(f"Seed: {SEED}")

    # Configure inference
    config = InferenceConfig(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        pipeline_config=str(MODEL_CONFIG),
        output_path=str(OUTPUT_DIR),
        seed=SEED,
        width=WIDTH,
        height=HEIGHT,
        num_frames=NUM_FRAMES,
        frame_rate=FRAME_RATE,
        offload_to_cpu=False,
        conditioning_media_paths=[str(IMAGE_FILE)],
        conditioning_strengths=[1.0],
        conditioning_start_frames=[0],
    )

    print("\nStarting LTX-Video inference...")
    print("Please wait. Model loading and generation may take time.")

    infer(config)

    print("\nPhase 4 completed.")
    print(f"Check generated video in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
