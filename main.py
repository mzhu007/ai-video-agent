
import argparse
import json
from pathlib import Path

from ollama import chat
from pydantic import BaseModel, Field, model_validator


# ============================================================
# 1. Schemas: define the contract for the AI-generated plan
# ============================================================

class ScenePlan(BaseModel):
    scene_number: int = Field(ge=1)
    scene_title: str

    location: str
    duration_seconds: int = Field(ge=1, le=60)

    visual_description: str
    subject: str
    environment: str

    time_of_day: str
    lighting: str
    atmosphere: str

    camera_shot: str
    camera_movement: str
    composition: str

    color_palette: list[str]

    video_prompt: str
    negative_prompt: str


class AudioPlan(BaseModel):
    music_style: str
    ambient_sounds: list[str]
    audio_description: str


class VideoPlan(BaseModel):
    title: str
    overall_style: str
    total_duration_seconds: int = Field(ge=1, le=60)
    aspect_ratio: str
    resolution: str

    scenes: list[ScenePlan]
    audio: AudioPlan


    @model_validator(mode="after")
    def validate_scene_durations(self):
        if not self.scenes:
            raise ValueError("At least one scene is required.")

        scene_duration = sum(
            scene.duration_seconds for scene in self.scenes
        )

        if scene_duration != self.total_duration_seconds:
            raise ValueError(
                f"Scene durations total {scene_duration}s, "
                f"but video duration is "
                f"{self.total_duration_seconds}s."
            )

        return self

    @model_validator(mode="after")
    def validate_aspect_ratio_and_resolution(self):
        try:
            width_str, height_str = self.resolution.strip().lower().split("x")
            width = int(width_str.strip())
            height = int(height_str.strip())
        except ValueError:
            raise ValueError(
                "Resolution must use the format WIDTHxHEIGHT, "
                "e.g. 1920x1080."
            )

        if width <= 0 or height <= 0:
            raise ValueError("Resolution dimensions must be positive.")

        supported_ratios = {
            "16:9": (16, 9),
            "9:16": (9, 16),
            "1:1": (1, 1),
        }

        if self.aspect_ratio not in supported_ratios:
            raise ValueError(
                f"Unsupported aspect ratio: {self.aspect_ratio}. "
                "Supported ratios: 16:9, 9:16, 1:1."
            )

        ratio_width, ratio_height = supported_ratios[self.aspect_ratio]

        if width * ratio_height != height * ratio_width:
            raise ValueError(
                f"Aspect ratio {self.aspect_ratio} does not match "
                f"resolution {self.resolution}."
            )

        return self

    @model_validator(mode="after")
    def validate_short_video_scenes(self):
        if self.total_duration_seconds <= 5 and len(self.scenes) != 1:
            raise ValueError(
                "Videos of 5 seconds or less must contain exactly one scene."
            )

        return self

    @model_validator(mode="after")
    def validate_scene_numbers(self):
        expected_numbers = list(range(1, len(self.scenes) + 1))

        actual_numbers = [
            scene.scene_number for scene in self.scenes
        ]

        if actual_numbers != expected_numbers:
            raise ValueError(
                f"Scene numbers must be sequential starting from 1. "
                f"Expected {expected_numbers}, got {actual_numbers}."
            )

        return self


# ============================================================
# 2. AI Director instructions
# ============================================================

SYSTEM_PROMPT = """
You are an expert cinematic film director and AI video production planner.

Convert the user's request into a structured video production plan.

Requirements:
- Respect the requested duration.
- For videos of 5 seconds or less, create exactly ONE scene.
- Scene durations must add up exactly to the total video duration.
- Prefer realistic scenery, natural camera movements, and coherent visuals.
- Describe observable visual details, not abstract concepts.
- Specify camera shot, movement, lighting, atmosphere, and composition.
- Generate a detailed video_prompt suitable for an AI video generation model.
- Include a negative_prompt to avoid common video artifacts.
- Use a coherent color palette.
- Do not invent dialogue or narration unless requested.
- Audio must complement the visual scene.
- Make the plan practical for AI video generation.
"""


# ============================================================
# 3. Generate and validate a plan
# ============================================================

def create_video_plan(
    user_request: str,
    model: str,
    max_attempts: int = 2,
) -> VideoPlan:

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_request,
        },
    ]

    for attempt in range(1, max_attempts + 1):

        print(f"Generating plan (attempt {attempt}/{max_attempts})...")

        response = chat(
            model=model,
            messages=messages,
            format=VideoPlan.model_json_schema(),
            options={"temperature": 0},
        )

        raw_output = response.message.content

        try:
            # Parse JSON and validate against our Pydantic schema
            plan = VideoPlan.model_validate_json(raw_output)

            print("Plan validated successfully!")
            return plan

        except Exception as exc:
            print(f"Validation failed: {exc}")

            # Give the model feedback and ask it to correct the result
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": raw_output,
                    },
                    {
                        "role": "user",
                        "content": (
                            "Your previous output failed validation. "
                            "Correct the output to satisfy the schema "
                            "and all requirements. Validation error: "
                            f"{exc}"
                        ),
                    },
                ]
            )

    raise RuntimeError(
        f"Could not generate a valid video plan "
        f"after {max_attempts} attempts."
    )


# ============================================================
# 4. Save the plan
# ============================================================

def save_plan(plan: VideoPlan, output_path: Path) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            plan.model_dump(),
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nPlan saved to: {output_path}")


# ============================================================
# 5. CLI entry point
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="AI Cinematic Video Director"
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        default=(
            "Make a 5-second cinematic video about "
            "autumn scenery in Wisconsin"
        ),
        help="Describe the video you want to create",
    )

    parser.add_argument(
        "--model",
        default="qwen3.5:9b",
        help="Ollama model to use",
    )

    parser.add_argument(
        "--output",
        default="output/video_plan.json",
        help="Output JSON file path",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("AI CINEMATIC VIDEO DIRECTOR")
    print("=" * 60)

    print(f"Model: {args.model}")
    print(f"Request: {args.prompt}")

    plan = create_video_plan(
        user_request=args.prompt,
        model=args.model,
    )

    print("\n" + "=" * 60)
    print("GENERATED VIDEO PLAN")
    print("=" * 60)

    print(plan.model_dump_json(indent=2))

    save_plan(
        plan=plan,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
