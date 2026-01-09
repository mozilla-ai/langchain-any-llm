"""Standard integration tests for ChatAnyLLM.

This module contains standard integration tests that verify ChatAnyLLM
conforms to the LangChain ChatModel interface.
"""

import os
from typing import Type

import pytest
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_anyllm import ChatAnyLLM


class TestChatAnyLLMStandard(ChatModelIntegrationTests):
    """Standard integration tests for ChatAnyLLM.

    This test class inherits from ChatModelIntegrationTests which provides
    a comprehensive suite of standard tests for LangChain chat models.

    Tests are opt-in by default. Override properties below to enable tests
    for specific features your integration supports.
    """

    @property
    def chat_model_class(self) -> Type[ChatAnyLLM]:
        """Return the ChatModel class to test."""
        return ChatAnyLLM

    @property
    def chat_model_params(self) -> dict:
        """Return initialization parameters for the chat model.

        Note: This should use a model that is available in the test environment.
        You may need to set appropriate environment variables (e.g., OPENAI_API_KEY).
        """
        # Use GPT-4o-mini for testing as it's fast and cost-effective
        # The test environment should have OPENAI_API_KEY set
        return {
            "model": "openai:gpt-4o-mini",
            "model_kwargs": {"temperature": 0},
        }

    @property
    def supports_image_inputs(self) -> bool:
        """Whether the model supports image inputs.

        any-llm supports image inputs for models that support it
        (e.g., GPT-4o, Claude 3, Gemini Pro Vision).

        Set to False for now as not all models support it and we're using
        a text-only model for testing.
        """
        return False

    @property
    def supports_anthropic_inputs(self) -> bool:
        """Whether the model supports Anthropic-style inputs.

        any-llm supports Anthropic models via the unified interface,
        but we're using OpenAI for testing, which doesn't support
        Anthropic-specific message formats (content_blocks, tool_use).
        """
        return False

    @property
    def supports_video_inputs(self) -> bool:
        """Whether the model supports video inputs.

        Not commonly supported yet.
        """
        return False

    @property
    def supports_audio_inputs(self) -> bool:
        """Whether the model supports audio inputs.

        Some models like GPT-4o support audio, but for general compatibility
        we'll set this to False.
        """
        return False

    @property
    def has_tool_calling(self) -> bool:
        """Whether the model supports tool/function calling.

        ChatAnyLLM supports tool calling via bind_tools.
        """
        return True

    @property
    def has_structured_output(self) -> bool:
        """Whether the model supports structured output.

        ChatAnyLLM supports structured output via with_structured_output.
        """
        return True

    @property
    def supports_json_mode(self) -> bool:
        """Whether the model supports JSON mode.

        Depends on the underlying model. OpenAI supports it, but not all models do.
        Set to False for general compatibility.
        """
        return False

    @property
    def returns_usage_metadata(self) -> bool:
        """Whether the model returns usage metadata.

        ChatAnyLLM returns usage metadata (token counts) in responses.
        """
        return True

    @property
    def supports_parallel_tool_calls(self) -> bool:
        """Whether the model supports parallel tool calls.

        Depends on the underlying model. OpenAI supports it, Anthropic added support.
        Set to True since we're using OpenAI for testing.
        """
        return True


# Skip tests if no API key is available
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set. "
    "These tests require a valid API key to run.",
)
