"""Test ChatAnyLLM chat model."""

import pytest

from langchain_anyllm import ChatAnyLLM


class TestChatAnyLLM:
    """Test ChatAnyLLM class."""

    def test_initialization(self) -> None:
        """Test ChatAnyLLM initialization."""
        llm = ChatAnyLLM(model="gpt-4", model_kwargs={"temperature": 0.5})
        assert llm.model == "gpt-4"
        assert llm.model_kwargs["temperature"] == 0.5
        assert llm._llm_type == "anyllm-chat"

    def test_initialization_with_first_class_params(self) -> None:
        """Test ChatAnyLLM initialization with first-class parameters."""
        llm = ChatAnyLLM(
            model="gpt-4",
            provider="openai",
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
            api_key="test-key",
            api_base="https://custom.api.com",
        )
        assert llm.model == "gpt-4"
        assert llm.provider == "openai"
        assert llm.temperature == 0.7
        assert llm.max_tokens == 100
        assert llm.top_p == 0.9
        assert llm.api_key == "test-key"
        assert llm.api_base == "https://custom.api.com"

    def test_default_params(self) -> None:
        """Test default parameters."""
        llm = ChatAnyLLM(
            model="gpt-3.5-turbo", model_kwargs={"temperature": 0.7, "max_tokens": 100}
        )
        params = llm._create_params()
        assert params["model"] == "gpt-3.5-turbo"
        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 100

        # Test with additional params
        llm2 = ChatAnyLLM(model="gpt-4", model_kwargs={"n": 2})
        params2 = llm2._create_params()
        assert params2["n"] == 2

    def test_first_class_params_in_create_params(self) -> None:
        """Test that first-class parameters are included in _create_params."""
        llm = ChatAnyLLM(
            model="gpt-4",
            provider="openai",
            temperature=0.5,
            max_tokens=200,
            top_p=0.95,
            response_format={"type": "json_object"},
        )
        params = llm._create_params()
        assert params["model"] == "gpt-4"
        assert params["provider"] == "openai"
        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 200
        assert params["top_p"] == 0.95
        assert params["response_format"] == {"type": "json_object"}

    def test_identifying_params_includes_provider(self) -> None:
        """Test that _identifying_params includes provider when set."""
        llm = ChatAnyLLM(model="gpt-4", provider="openai")
        params = llm._identifying_params
        assert params["model"] == "gpt-4"
        assert params["provider"] == "openai"

        # Without provider
        llm2 = ChatAnyLLM(model="openai:gpt-4")
        params2 = llm2._identifying_params
        assert params2["model"] == "openai:gpt-4"
        assert "provider" not in params2

    @pytest.mark.asyncio
    async def test_async_initialization(self) -> None:
        """Test async functionality exists."""
        llm = ChatAnyLLM(model="gpt-4")
        # Just verify the async methods exist
        assert hasattr(llm, "ainvoke")
        assert hasattr(llm, "astream")
        assert hasattr(llm, "_agenerate")

    def test_bind_tools(self) -> None:
        """Test binding tools to model."""
        llm = ChatAnyLLM(model="gpt-4")

        def dummy_tool(x: int) -> int:
            """Dummy tool."""
            return x * 2

        llm_with_tools = llm.bind_tools([dummy_tool])
        assert llm_with_tools is not None

    def test_with_structured_output(self) -> None:
        """Test structured output binding."""
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            """Test schema."""

            name: str
            age: int

        llm = ChatAnyLLM(model="gpt-4")
        structured_llm = llm.with_structured_output(TestSchema)
        assert structured_llm is not None
