import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from groq import GroqError

import run
from app.application import create_app
from app.config import (
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    Settings,
)


UNKNOWN_ANSWER = "I can't answer this question since I have no knowledge about it"


class AskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.knowledge_file = root / "knowledge_base.txt"
        self.prompt_file = root / "base_prompt.txt"
        self.knowledge_file.write_text("BEON.tech offers QA services.", encoding="utf-8")
        self.prompt_file.write_text(f"Use only the knowledge. Otherwise say: {UNKNOWN_ANSWER}", encoding="utf-8")
        self.settings = Settings(
            api_key="test-key",
            knowledge_file=self.knowledge_file,
            prompt_file=self.prompt_file,
        )
        self.client = TestClient(create_app(self.settings))

    def mock_groq_answer(self, answer: str):
        groq_patcher = patch("app.services.qa.Groq")
        mock_groq = groq_patcher.start()
        self.addCleanup(groq_patcher.stop)
        mock_groq.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=answer))]
        )
        return mock_groq

    def test_versioned_question_uses_settings_and_returns_json(self) -> None:
        mock_groq = self.mock_groq_answer("BEON.tech offers QA services.")

        response = self.client.post("/api/v1/ask", json={"question": "What does BEON.tech offer?"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"answer": "BEON.tech offers QA services."})
        self.assertEqual(self.client.post("/ask", json={"question": "Hi"}).status_code, 404)
        call = mock_groq.return_value.chat.completions.create.call_args
        self.assertEqual(call.kwargs["model"], DEFAULT_MODEL)
        self.assertEqual(call.kwargs["temperature"], DEFAULT_TEMPERATURE)
        self.assertEqual(call.kwargs["max_completion_tokens"], DEFAULT_MAX_COMPLETION_TOKENS)
        self.assertIn("QA services", call.kwargs["messages"][1]["content"])
        self.assertIn("What does BEON.tech offer?", call.kwargs["messages"][1]["content"])

    def test_unrelated_question_uses_prompt_fallback(self) -> None:
        mock_groq = self.mock_groq_answer(UNKNOWN_ANSWER)

        response = self.client.post("/api/v1/ask", json={"question": "What is the weather?"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"answer": UNKNOWN_ANSWER})
        call = mock_groq.return_value.chat.completions.create.call_args
        self.assertIn(UNKNOWN_ANSWER, call.kwargs["messages"][0]["content"])

    def test_blank_and_missing_questions_are_rejected(self) -> None:
        mock_groq = self.mock_groq_answer("unused")

        for body in ({"question": "   "}, {}):
            with self.subTest(body=body):
                self.assertEqual(self.client.post("/api/v1/ask", json=body).status_code, 422)
        mock_groq.assert_not_called()

    def test_both_text_files_are_reread_for_each_request(self) -> None:
        mock_groq = self.mock_groq_answer("answer")

        self.client.post("/api/v1/ask", json={"question": "What services?"})
        self.knowledge_file.write_text("BEON.tech offers DevOps services.", encoding="utf-8")
        self.prompt_file.write_text("New concise prompt.", encoding="utf-8")
        self.client.post("/api/v1/ask", json={"question": "What services?"})

        calls = mock_groq.return_value.chat.completions.create.call_args_list
        self.assertIn("QA services", calls[0].kwargs["messages"][1]["content"])
        self.assertIn("DevOps services", calls[1].kwargs["messages"][1]["content"])
        self.assertIn("Use only the knowledge", calls[0].kwargs["messages"][0]["content"])
        self.assertEqual(calls[1].kwargs["messages"][0]["content"], "New concise prompt.")

    def test_health_does_not_need_key_or_sources(self) -> None:
        missing = Path(self.temp_dir.name) / "missing.txt"
        client = TestClient(create_app(Settings(api_key=None, knowledge_file=missing, prompt_file=missing)))

        self.assertEqual(client.get("/api/v1/health").json(), {"status": "ok"})
        self.assertEqual(client.post("/api/v1/ask", json={"question": "Hi"}).status_code, 503)

    def test_groq_failure_returns_service_error_without_key(self) -> None:
        mock_groq = self.mock_groq_answer("unused")
        mock_groq.return_value.chat.completions.create.side_effect = GroqError("failure")

        response = self.client.post("/api/v1/ask", json={"question": "What services?"})

        self.assertEqual(response.status_code, 502)
        self.assertNotIn("test-key", response.text)


class SettingsTests(unittest.TestCase):
    def test_defaults_and_environment_overrides(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            defaults = Settings.from_environment()
        self.assertEqual((defaults.model, defaults.temperature, defaults.max_completion_tokens),
                         (DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_COMPLETION_TOKENS))

        with patch.dict(os.environ, {
            "GROQ_MODEL": "custom-model",
            "GROQ_TEMPERATURE": "0.4",
            "GROQ_MAX_COMPLETION_TOKENS": "512",
        }, clear=True):
            settings = Settings.from_environment()
        self.assertEqual((settings.model, settings.temperature, settings.max_completion_tokens),
                         ("custom-model", 0.4, 512))

    def test_invalid_environment_settings_fail_at_startup(self) -> None:
        for values in ({"GROQ_TEMPERATURE": "3"}, {"GROQ_MAX_COMPLETION_TOKENS": "0"}):
            with self.subTest(values=values), patch.dict(os.environ, values, clear=True):
                with self.assertRaises(ValueError):
                    Settings.from_environment()


class RunnerTests(unittest.TestCase):
    def test_cli_arguments_override_environment(self) -> None:
        with patch.dict(os.environ, {
            "GROQ_MODEL": "env-model",
            "GROQ_TEMPERATURE": "0.1",
            "GROQ_MAX_COMPLETION_TOKENS": "128",
        }, clear=True), patch("run.uvicorn.run") as uvicorn_run:
            run.main(["--model", "cli-model", "--temperature", "0.2",
                      "--max-completion-tokens", "384", "--reload"])
            settings = Settings.from_environment()

        self.assertEqual((settings.model, settings.temperature, settings.max_completion_tokens),
                         ("cli-model", 0.2, 384))
        uvicorn_run.assert_called_once_with("main:app", host="127.0.0.1", port=8000, reload=True)

    def test_invalid_cli_values_exit_before_starting_server(self) -> None:
        for arguments in (["--temperature", "3"], ["--max-completion-tokens", "0"]):
            with self.subTest(arguments=arguments), patch("run.uvicorn.run") as uvicorn_run:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit_error:
                    run.main(arguments)
                self.assertEqual(exit_error.exception.code, 2)
                uvicorn_run.assert_not_called()

    def test_invalid_environment_stops_before_starting_server(self) -> None:
        with patch.dict(os.environ, {"GROQ_TEMPERATURE": "3"}, clear=True):
            with patch("run.uvicorn.run") as uvicorn_run:
                with self.assertRaises(ValueError):
                    run.main([])
                uvicorn_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
