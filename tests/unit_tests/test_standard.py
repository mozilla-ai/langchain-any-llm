"""Standard unit tests for ChatAnyLLM.

This module contains standard unit tests that verify ChatAnyLLM
follows the LangChain ChatModel interface without making API calls.
"""

from typing import Type

from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_anyllm import ChatAnyLLM


class TestChatAnyLLMUnit(ChatModelUnitTests):
    """Standard unit tests for ChatAnyLLM.

    This test class inherits from ChatModelUnitTests which provides
    a comprehensive suite of unit tests for LangChain chat models.

    Unit tests do not make external API calls and verify basic
    functionality like initialization, serialization, etc.
    """

    @property
    def chat_model_class(self) -> Type[ChatAnyLLM]:
        """Return the ChatModel class to test."""
        return ChatAnyLLM

    @property
    def chat_model_params(self) -> dict:
        """Return initialization parameters for the chat model.

        For unit tests, we use a dummy model name since no API calls are made.
        """
        return {
            "model": "openai:gpt-4o-mini",
        }

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        """Parameters for testing initialization from environment variables.

        Returns a tuple of (init_params, expected_params, env_vars).
        """
        return (
            {},  # init params
            {"model": "openai:gpt-4o-mini"},  # expected params after init
            {},  # environment variables to set
        )
