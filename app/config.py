import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_COMPLETION_TOKENS = 256

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS
    knowledge_file: Path = PROJECT_ROOT / "knowledge_base.txt"
    prompt_file: Path = PROJECT_ROOT / "app" / "prompts" / "base_prompt.txt"

    def __post_init__(self) -> None:
        if not 0 <= self.temperature <= 2:
            raise ValueError("GROQ_TEMPERATURE must be between 0 and 2")
        if self.max_completion_tokens <= 0:
            raise ValueError("GROQ_MAX_COMPLETION_TOKENS must be positive")

    @classmethod
    def from_environment(cls) -> "Settings":
        try:
            temperature = float(os.getenv("GROQ_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
            max_tokens = int(
                os.getenv("GROQ_MAX_COMPLETION_TOKENS", str(DEFAULT_MAX_COMPLETION_TOKENS))
            )
        except ValueError as exc:
            raise ValueError("Invalid Groq generation setting") from exc

        return cls(
            api_key=os.getenv("GROQ_API_KEY"),
            model=os.getenv("GROQ_MODEL") or DEFAULT_MODEL,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
