from pydantic import BaseModel, field_validator


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("Question must not be blank")
        return question


class AskResponse(BaseModel):
    answer: str
