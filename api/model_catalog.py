from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelSpec:
    public_id: str
    provider: str
    operation: str
    upstream_id: str
    base_rpm: int
    base_tpm: Optional[int] = None


GEMMA_MODEL = "gemma-4-31b"
GLM_MODEL = "zai-glm-4.7"
GPT_MODEL = "gpt-oss-120b"

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b",
    "qwen/qwen3.6-27b",
]

AGNES_TEXT_MODEL = "agnes/agnes-2.5-flash"
AGNES_IMAGE_MODEL = "agnes/agnes-image-2.1-flash"
AGNES_VIDEO_MODEL = "agnes/agnes-video-v2.0"

CEREBRAS_MODELS = [GEMMA_MODEL, GLM_MODEL, GPT_MODEL]
AGNES_MODELS = [AGNES_TEXT_MODEL, AGNES_IMAGE_MODEL, AGNES_VIDEO_MODEL]

MODEL_CATALOG: Dict[str, ModelSpec] = {
    model: ModelSpec(model, "cerebras", "chat", model, 5, 30000)
    for model in CEREBRAS_MODELS
}
MODEL_CATALOG.update({
    model: ModelSpec(model, "groq", "chat", model, 30, 40000)
    for model in GROQ_MODELS
})
MODEL_CATALOG.update({
    AGNES_TEXT_MODEL: ModelSpec(AGNES_TEXT_MODEL, "agnes", "chat", "agnes-2.5-flash", 20),
    AGNES_IMAGE_MODEL: ModelSpec(AGNES_IMAGE_MODEL, "agnes", "image", "agnes-image-2.1-flash", 20),
    AGNES_VIDEO_MODEL: ModelSpec(AGNES_VIDEO_MODEL, "agnes", "video", "agnes-video-v2.0", 1),
})


def get_model_spec(model: str) -> Optional[ModelSpec]:
    return MODEL_CATALOG.get(model)


def models_for_provider(provider: str) -> List[str]:
    return [spec.public_id for spec in MODEL_CATALOG.values() if spec.provider == provider]


def models_for_operation(operation: str) -> List[str]:
    return [spec.public_id for spec in MODEL_CATALOG.values() if spec.operation == operation]
