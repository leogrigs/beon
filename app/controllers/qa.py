from fastapi import APIRouter, HTTPException, Request

from app.schemas.qa import AskRequest, AskResponse
from app.services.qa import (
    GroqAnswerError,
    MissingAPIKeyError,
    SourceUnavailableError,
    answer_question,
)


router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    try:
        answer = answer_question(payload.question, request.app.state.settings)
    except MissingAPIKeyError as exc:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured") from exc
    except SourceUnavailableError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except GroqAnswerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return AskResponse(answer=answer)
