import json
from pathlib import Path

import ollama
from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL = "qwen3.5:9b"

PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "output"

INPUT_FILE = OUTPUT_DIR / "video_plan.json"
OUTPUT_FILE = OUTPUT_DIR / "cinematic_prompts.json"


# --------------------------------------------------
# Phase 2 output schema
# --------------------------------------------------

class VideoPlanScene(BaseModel):
    scene_number: int = Field(
        ge=1,
        description="Scene number, starting at 1"
    )

    duration_seconds: int = Field(
        gt=0,
        description="Duration of this scene in seconds"
    )

class VideoPlan(BaseModel):
    scenes: list[VideoPlanScene] = Field(
        min_length=1
    )

    total_duration_seconds: int = Field(
        gt=0,
        description="Total video duration in seconds"
    )

    @model_validator(mode="after")
    def validate_total_duration(self):
        actual_total = sum(
            scene.duration_seconds
            for scene in self.scenes
        )

        if actual_total != self.total_duration_seconds:
            raise ValueError(
                f"Total duration mismatch: "
                f"Expected {self.total_duration_seconds}, "
                f"but scene durations sum to {actual_total}"
            )

        return self

    @model_validator(mode="after")
    def validate_scene_numbers(self):
        expected_numbers = list(
            range(1, len(self.scenes) + 1)
        )

        actual_numbers = [
            scene.scene_number
            for scene in self.scenes
        ]

        if actual_numbers != expected_numbers:
            raise ValueError(
                f"Scene numbers must be sequential starting at 1. "
                f"Expected {expected_numbers}, "
                f"but got {actual_numbers}"
            )

        return self
    
class CinematicScenePrompt(BaseModel):
    scene_number: int = Field(
        ge=1,
        description="Scene number, starting at 1"
    )

    duration_seconds: int = Field(
        gt=0,
        description="Duration of this scene in seconds"
    )

    positive_prompt: str = Field(
        min_length=20,
        description="Detailed cinematic prompt for video generation"
    )

    negative_prompt: str = Field(
        min_length=5,
        description="Things to avoid when generating the video"
    )

    camera_direction: str = Field(
        min_length=5,
        description="Camera position, lens feel, and movement"
    )

    visual_style: str = Field(
        min_length=5,
        description="Visual style, lighting, color grading, realism"
    )

class CinematicPromptPlan(BaseModel):
    scenes: list[CinematicScenePrompt] = Field(
        min_length=1,
        description="List of cinematic prompts, one for each video scene"
    )

# --------------------------------------------------
# Prompt engineering
# --------------------------------------------------

SYSTEM_PROMPT = """
You are an expert cinematographer, film director, and AI video prompt engineer.

Your job is to transform a high-level video plan into production-ready
prompts for AI video generation models.

Preserve the original creative intent, subject, location, environment,
and duration of each scene.

For every scene:

1. Create a detailed positive video-generation prompt.
2. Specify the subject and environmental details.
3. Describe lighting and atmosphere.
4. Describe composition and depth.
5. Specify camera position and movement.
6. Describe realistic motion over time.
7. Keep the scene visually coherent.
8. Create a negative prompt to avoid visual artifacts.
9. Avoid introducing unrelated objects or events.
10. Avoid contradictory instructions.
11. Preserve the original subject and environment from the input plan.
    Do not replace a country road with a forest path, or vice versa.
12. Maintain a smooth, stabilized camera unless handheld movement
    is explicitly requested.
13. Use horizontal 16:9 composition consistently.
    Avoid contradictory instructions such as "vertical or horizontal".
14. Maintain continuity between consecutive scenes.
    Preserve consistent environmental conditions, lighting,
    weather, time of day, and visual characteristics unless
    the original video plan explicitly specifies a change.

The positive prompt should be written as natural language suitable for
a text-to-video or image-to-video model.

The output must follow the supplied JSON schema.
Do not include markdown or explanations outside the JSON.
"""


USER_PROMPT_TEMPLATE = """
Transform the following video plan into cinematic video-generation prompts.

VIDEO PLAN:
{video_plan}

Requirements:

- Preserve the number of scenes.
- Preserve scene order.
- Preserve each scene's duration.
- Do not invent additional scenes.
- Make each prompt visually specific.
- Include realistic motion appropriate for the duration.
- Avoid excessive camera movements.
- Prefer smooth, physically plausible motion.
- Keep the visual style consistent across scenes.
- Preserve the original scene's main subject exactly.
- Do not creatively replace important objects or locations.
- Use smooth, stabilized camera movements.
- Assume landscape orientation (16:9).

Return one structured prompt for each original scene.
"""


# --------------------------------------------------
# Generate cinematic prompts
# --------------------------------------------------

def generate_cinematic_prompts(
    video_plan: VideoPlan,
) -> CinematicPromptPlan:
    
    video_plan_json = json.dumps(
        video_plan.model_dump(),
        indent=2,
        ensure_ascii=False,
    )

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    video_plan=video_plan_json
                ),
            },
        ],
        format=CinematicPromptPlan.model_json_schema(),

        # Disable internal reasoning for this prompt-generation task.
        think=False,        

        options={
            "temperature": 0.7,
            "num_predict": 8192,
        },
    )

    content = response.message.content or ""

    # Save raw response for debugging.
    debug_file = OUTPUT_DIR / "phase2_raw_response.txt"

    debug_file.write_text(
        content,
        encoding="utf-8",
    )

    print(f"Raw response saved to: {debug_file}")
    print(f"Generated tokens: {getattr(response, 'eval_count', None)}")
    print(f"Done reason: {getattr(response, 'done_reason', None)}")

    done_reason = getattr(response, "done_reason", None)

    if done_reason == "length":
        raise RuntimeError(
            "Ollama stopped generation because it reached "
            "the maximum output token limit. "
            "Increase num_predict and try again."
        )

    # Guard against empty output before attempting JSON validation.
    if not content.strip():
        thinking = getattr(response.message, "thinking", None)

        if thinking:
            print("\nThe model generated thinking but no final content.")
            print("Thinking output:")
            print(thinking[:1000])

        raise RuntimeError(
            "Ollama returned empty content. "
            "The model did not generate a final response."
        )

    prompt_plan = CinematicPromptPlan.model_validate_json(
        content
    )

    # --------------------------------------------------
    # Additional validation against Phase 1
    # --------------------------------------------------

    original_scenes = video_plan.scenes
    generated_scenes = prompt_plan.scenes

    if len(original_scenes) != len(generated_scenes):
        raise ValueError(
            "Scene count mismatch: "
            f"Phase 1={len(original_scenes)}, "
            f"Phase 2={len(generated_scenes)}"
        )

    for index, (original, generated) in enumerate(
        zip(original_scenes, generated_scenes),
        start=1,
    ):
        expected_scene_number = original.scene_number

        if generated.scene_number != expected_scene_number:
            raise ValueError(
                f"Scene number mismatch in scene {index}: "
                f"Expected {expected_scene_number}, "
                f"Got {generated.scene_number}"
            )
    
        if generated.duration_seconds != original.duration_seconds:
            raise ValueError(
                f"Duration mismatch in scene {index}"
            )

    original_total = video_plan.total_duration_seconds

    generated_total = sum(
        scene.duration_seconds
        for scene in generated_scenes
    )

    if original_total != generated_total:
        raise ValueError(
            f"Total duration mismatch: "
            f"{original_total} != {generated_total}"
        )

    return prompt_plan


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("Phase 2: Cinematic Prompt Engineer")
    print(f"Model: {MODEL}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load Phase 1 output.
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        video_plan = VideoPlan.model_validate_json(
            file.read()
        )

    print(f"Loaded video plan: {INPUT_FILE}")

    # Generate cinematic prompts.
    prompt_plan = generate_cinematic_prompts(
        video_plan
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            prompt_plan.model_dump(),
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nPhase 2 completed successfully!")
    print(f"Output saved to: {OUTPUT_FILE}")

    # Display results.
    for scene in prompt_plan.scenes:

        print(f"\n{'=' * 60}")
        print(f"Scene {scene.scene_number}")
        print(f"Duration: {scene.duration_seconds}s")

        print("\nPositive prompt:")
        print(scene.positive_prompt)

        print("\nNegative prompt:")
        print(scene.negative_prompt)

        print("\nCamera:")
        print(scene.camera_direction)

        print("\nVisual style:")
        print(scene.visual_style)


if __name__ == "__main__":
    main()