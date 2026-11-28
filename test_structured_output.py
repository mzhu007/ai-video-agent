from ollama import chat
from pydantic import BaseModel


class Scene(BaseModel):
    location: str
    camera: str
    lighting: str


response = chat(
    model="qwen3.5:9b",
    messages=[
        {
            "role": "user",
            "content": (
                "Describe a cinematic 5-second Wisconsin autumn scene."
            ),
        }
    ],
    format=Scene.model_json_schema(),
    options={"temperature": 0},
)

scene = Scene.model_validate_json(response.message.content)

print("\nStructured output:")
print(scene.model_dump_json(indent=2))