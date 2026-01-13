"""Standard unit tests for ChatAnyLLM.

This module contains standard unit tests that verify ChatAnyLLM
follows the LangChain ChatModel interface without making API calls.
"""

from typing import Type

from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_anyllm import ChatAnyLLM


class TestChatAnyLLMUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatAnyLLM]:
        """Return the ChatModel class to test."""
        return ChatAnyLLM

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "openai:gpt-4o-mini",
        }

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {},  # init params
            {"model": "openai:gpt-4o-mini"},  # expected params after init
            {},  # environment variables to set
        )
