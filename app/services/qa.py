from pathlib import Path

from groq import Groq, GroqError

from app.config import Settings


class MissingAPIKeyError(Exception):
    pass


class SourceUnavailableError(Exception):
    pass


class GroqAnswerError(Exception):
    pass


def read_source(path: Path, label: str) -> str:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SourceUnavailableError(f"{label} is unavailable") from exc
    if not content.strip():
        raise SourceUnavailableError(f"{label} is empty")
    return content.strip()


def answer_question(question: str, settings: Settings) -> str:
    if not settings.api_key:
        raise MissingAPIKeyError

    prompt = read_source(settings.prompt_file, "Base prompt")
    knowledge = read_source(settings.knowledge_file, "Knowledge base")

    try:
        completion = Groq(api_key=settings.api_key, timeout=30.0).chat.completions.create(
            model=settings.model,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"Knowledge base:\n<knowledge>\n{knowledge}\n</knowledge>\n\n"
                        f"Question: {question}"
                    ),
                },
            ],
            temperature=settings.temperature,
            max_completion_tokens=settings.max_completion_tokens,
        )
    except GroqError as exc:
        raise GroqAnswerError("Groq could not generate an answer") from exc

    answer = completion.choices[0].message.content
    if not answer or not answer.strip():
        raise GroqAnswerError("Groq returned an empty answer")
    return answer.strip()
