import argparse
import os

import uvicorn


def temperature_arg(value: str) -> float:
    try:
        temperature = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("temperature must be a number") from exc
    if not 0 <= temperature <= 2:
        raise argparse.ArgumentTypeError("temperature must be between 0 and 2")
    return temperature


def positive_int_arg(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("max completion tokens must be an integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("max completion tokens must be positive")
    return number


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the BEON.tech Q&A API")
    parser.add_argument("--model", help="Groq model ID")
    parser.add_argument("--temperature", type=temperature_arg)
    parser.add_argument("--max-completion-tokens", type=positive_int_arg)
    parser.add_argument("--reload", action="store_true", help="reload Python code during development")
    args = parser.parse_args(argv)

    if args.model is not None:
        if not args.model.strip():
            parser.error("model must not be blank")
        os.environ["GROQ_MODEL"] = args.model.strip()
    if args.temperature is not None:
        os.environ["GROQ_TEMPERATURE"] = str(args.temperature)
    if args.max_completion_tokens is not None:
        os.environ["GROQ_MAX_COMPLETION_TOKENS"] = str(args.max_completion_tokens)

    from app.config import Settings

    Settings.from_environment()
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=args.reload)


if __name__ == "__main__":
    main()
