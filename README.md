# BEON.tech Q&A starter

A small FastAPI service that answers questions using a local text knowledge base and Groq. Each request reads `knowledge_base.txt` and `app/prompts/base_prompt.txt` again, so text edits take effect on the next request.

## Requirements

- Python 3.10 or newer
- A Groq API key and network access to Groq
- The dependencies in `requirements.txt`

The service is instructed to answer only from the three BEON.tech facts in `knowledge_base.txt`. For an unrelated question, the prompt asks it to say: “I can't answer this question since I have no knowledge about it”.

## Layout

- `main.py`: FastAPI entry point
- `run.py`: startup arguments and Uvicorn launcher
- `app/controllers/`: HTTP routes
- `app/services/`: knowledge and prompt loading, plus Groq calls
- `app/schemas/`: request and response models
- `app/config.py`: validated settings
- `app/prompts/base_prompt.txt`: editable system prompt

## Setup and run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the key as an environment variable:

```bash
export GROQ_API_KEY="your-groq-api-key"
```

Alternatively, put `GROQ_API_KEY=your-groq-api-key` in a `.env` file beside `main.py`. The `.env` file is ignored by Git.

Start the API:

```bash
python run.py --reload
```

The default model is `openai/gpt-oss-20b`, the default temperature is `0`, and the default maximum completion length is `256` tokens. Override them when starting the API:

```bash
python run.py --model openai/gpt-oss-20b --temperature 0.2 --max-completion-tokens 384 --reload
```

Settings take precedence in this order: startup arguments, environment variables, `.env`, built-in defaults. The corresponding variable names are `GROQ_MODEL`, `GROQ_TEMPERATURE`, and `GROQ_MAX_COMPLETION_TOKENS`. Temperature must be between 0 and 2; the token limit must be positive. You can also run `uvicorn main:app --reload` when using environment variables or `.env` instead of startup arguments.

Send a question:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What services does BEON.tech offer?"}'
```

The response is JSON in the form `{"answer":"..."}`. `GET /api/v1/health` returns `{"status":"ok"}` without checking Groq or the knowledge file. Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

Missing or blank questions return HTTP 422. A missing API key returns HTTP 503; Groq failures return HTTP 502. The service does not include the API key in error responses.

## Checks

Run the automated checks without making a Groq request:

```bash
python -m unittest -v
```

With a real API key configured, the `curl` example above is an optional live check. You can also ask an unrelated question and confirm the fallback answer.
