"""AnyLLM chat model integration for LangChain.

This module provides a LangChain-compatible chat model wrapper for AnyLLM,
enabling seamless integration with LangChain's ecosystem.
"""

from __future__ import annotations

import logging
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    Sequence,
)

from any_llm import acompletion, completion
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field

from langchain_anyllm.utils import (
    _convert_delta_to_message_chunk,
    _convert_dict_to_message,
    _convert_message_to_dict,
)

logger = logging.getLogger(__name__)


class ChatAnyLLM(BaseChatModel):
    """Chat model that uses the AnyLLM API."""

    model: str
    api_key: str | None = None
    api_base: str | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    stream_options: dict[str, Any] | None = Field(
        default_factory=lambda: {"include_usage": True}
    )

    def _is_anthropic_model(self) -> bool:
        """Check if the model is an Anthropic model."""
        return self.model.startswith("anthropic:") or "claude" in self.model.lower()

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatResult:
        if stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        message_dicts = [_convert_message_to_dict(m) for m in messages]
        logger.warning(f"Message dicts: {message_dicts}")
        params = self._create_params(stop, **kwargs)
        response = completion(messages=message_dicts, **params)  # type: ignore[arg-type]
        if not isinstance(response, ChatCompletion):
            error_message = f"Expected ChatCompletion, got {type(response)}"
            raise ValueError(error_message)
        return self._create_chat_result(response)

    def _create_chat_result(self, response: ChatCompletion) -> ChatResult:
        resp_dict = response.model_dump()

        generations = []
        token_usage = response.usage
        for res in resp_dict["choices"]:
            message = _convert_dict_to_message(res["message"])
            if isinstance(message, AIMessage) and token_usage:
                message.response_metadata = {"model_name": self.model}
                message.usage_metadata = UsageMetadata(
                    input_tokens=token_usage.prompt_tokens,
                    output_tokens=token_usage.completion_tokens,
                    total_tokens=token_usage.prompt_tokens
                    + token_usage.completion_tokens,
                )
            gen = ChatGeneration(
                message=message,
                generation_info={"finish_reason": res.get("finish_reason")},
            )
            generations.append(gen)

        llm_output = {
            "token_usage": token_usage,
            "model": self.model,
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _create_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        params = {
            "api_key": self.api_key,
            "api_base": self.api_base,
            "model": self.model,
            **self.model_kwargs,
        }

        is_anthropic = self._is_anthropic_model()

        # Handle stop sequences - Anthropic uses stop_sequences instead of stop
        if stop is not None:
            if is_anthropic:
                if "stop_sequences" in params:
                    error_message = (
                        "`stop_sequences` found in both the input and default params."
                    )
                    raise ValueError(error_message)
                params["stop_sequences"] = stop
            else:
                if "stop" in params:
                    error_message = "`stop` found in both the input and default params."
                    raise ValueError(error_message)
                params["stop"] = stop

        # Translate LangChain tool_choice to OpenAI-compatible values
        # Only include tool_choice if tools are present
        if "tool_choice" in kwargs and "tools" in kwargs:
            tool_choice = kwargs["tool_choice"]
            if tool_choice == "any":
                # LangChain uses 'any', OpenAI uses 'required'
                params["tool_choice"] = "required"
            elif tool_choice is True:
                params["tool_choice"] = "required"
            elif tool_choice is False:
                params["tool_choice"] = "none"
            elif isinstance(tool_choice, str) and tool_choice not in [
                "none",
                "auto",
                "required",
            ]:
                # If it's a string that's not a standard value,
                # treat it as a function name
                params["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice},
                }
            else:
                # Pass through dicts and standard string values
                params["tool_choice"] = tool_choice

        # Pass through all kwargs except our special handling
        for key, value in kwargs.items():
            if key not in ["tool_choice"]:
                params[key] = value

        return params

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        params = self._create_params(stop, **kwargs)
        params["stream"] = True

        # Set stream_options for usage metadata if not already provided
        # Note: Anthropic doesn't support stream_options parameter
        if not self._is_anthropic_model():
            if "stream_options" not in params and self.stream_options:
                params["stream_options"] = self.stream_options

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        result = completion(messages=message_dicts, **params)  # type: ignore[arg-type]

        if not isinstance(result, Iterator):
            error_message = f"Expected Iterator, got {type(result)}"
            raise ValueError(error_message)

        # Iterate over stream results
        for chunk_item in result:
            chunk_dict: dict[str, Any] = chunk_item.model_dump()

            # Handle usage-only chunk (final chunk with empty choices but usage data)
            if len(chunk_dict["choices"]) == 0:
                if chunk_dict.get("usage"):
                    # Create an empty chunk with usage metadata
                    usage = chunk_dict["usage"]
                    usage_chunk = AIMessageChunk(
                        content="",
                        response_metadata={"model_name": self.model},
                        usage_metadata=UsageMetadata(
                            input_tokens=usage.get("prompt_tokens", 0),
                            output_tokens=usage.get("completion_tokens", 0),
                            total_tokens=usage.get("total_tokens", 0),
                        ),
                    )
                    cg_chunk = ChatGenerationChunk(message=usage_chunk)
                    yield cg_chunk
                continue

            choice = chunk_dict["choices"][0]
            delta = choice["delta"]

            message_chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)

            # Only set usage_metadata on the final chunk (when finish_reason is set)
            finish_reason = choice.get("finish_reason")
            if finish_reason and chunk_dict.get("usage"):
                if isinstance(message_chunk, AIMessageChunk):
                    usage = chunk_dict["usage"]
                    message_chunk.usage_metadata = UsageMetadata(
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                        total_tokens=usage.get("total_tokens", 0),
                    )
                    message_chunk.response_metadata = {"model_name": self.model}
            default_chunk_class = message_chunk.__class__
            cg_chunk = ChatGenerationChunk(message=message_chunk)
            if run_manager:
                content = message_chunk.content
                if isinstance(content, str):
                    run_manager.on_llm_new_token(content, chunk=cg_chunk)
            yield cg_chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        params = self._create_params(stop, **kwargs)
        params["stream"] = True

        # Set stream_options for usage metadata if not already provided
        # Note: Anthropic doesn't support stream_options parameter
        if not self._is_anthropic_model():
            if "stream_options" not in params and self.stream_options:
                params["stream_options"] = self.stream_options

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        result = await acompletion(messages=message_dicts, **params)  # type: ignore[arg-type]
        if not isinstance(result, AsyncIterator):
            error_message = f"Expected AsyncIterator, got {type(result)}"
            raise ValueError(error_message)
        async for stream_chunk in result:
            if not isinstance(stream_chunk, ChatCompletionChunk):
                error_message = "Unexpected chunk type"
                raise ValueError(error_message)

            # Handle usage-only chunk (final chunk with empty choices but usage data)
            if len(stream_chunk.choices) == 0:
                if hasattr(stream_chunk, "usage") and stream_chunk.usage:
                    usage = stream_chunk.usage.model_dump()
                    usage_chunk = AIMessageChunk(
                        content="",
                        response_metadata={"model_name": self.model},
                        usage_metadata=UsageMetadata(
                            input_tokens=usage.get("prompt_tokens", 0),
                            output_tokens=usage.get("completion_tokens", 0),
                            total_tokens=usage.get("total_tokens", 0),
                        ),
                    )
                    cg_chunk = ChatGenerationChunk(message=usage_chunk)
                    yield cg_chunk
                continue

            for choice in stream_chunk.choices:
                delta = choice.delta
                message_chunk = _convert_delta_to_message_chunk(
                    delta, default_chunk_class
                )

                # Only set usage_metadata on the final chunk (when finish_reason is set)
                if choice.finish_reason:
                    if hasattr(stream_chunk, "usage") and stream_chunk.usage:
                        if isinstance(message_chunk, AIMessageChunk):
                            usage = stream_chunk.usage.model_dump()
                            message_chunk.usage_metadata = UsageMetadata(
                                input_tokens=usage.get("prompt_tokens", 0),
                                output_tokens=usage.get("completion_tokens", 0),
                                total_tokens=usage.get("total_tokens", 0),
                            )
                            message_chunk.response_metadata = {"model_name": self.model}
                default_chunk_class = message_chunk.__class__
                cg_chunk = ChatGenerationChunk(message=message_chunk)
                if run_manager:
                    content = message_chunk.content
                    if isinstance(content, str):
                        await run_manager.on_llm_new_token(content, chunk=cg_chunk)
                yield cg_chunk

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        stream: bool | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else False
        if should_stream:
            stream_iter = self._astream(
                messages=messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts = [_convert_message_to_dict(m) for m in messages]
        params = self._create_params(stop, **kwargs)
        response = await acompletion(messages=message_dicts, **params)  # type: ignore[arg-type]
        if not isinstance(response, ChatCompletion):
            error_message = f"Expected ChatCompletion, got {type(response)}"
            raise ValueError(error_message)
        return self._create_chat_result(response)

    def bind_tools(
        self,
        tools: Sequence[
            dict[str, Any] | type[BaseModel] | Callable[..., Any] | BaseTool
        ],
        tool_choice: dict[str, Any] | str | bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tool-like objects to this chat model."""
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, tool_choice=tool_choice, **kwargs)

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            **self.model_kwargs,
        }

    @property
    def _llm_type(self) -> str:
        return "anyllm-chat"
