"""Utility functions for message conversion.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolCallChunk,
    ToolMessage,
)

logger = logging.getLogger(__name__)


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: Dictionary containing message data with 'role' and 'content' keys

    Returns:
        Appropriate BaseMessage subclass based on role
    """
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    if role == "assistant":
        # Handle content - can be string, None, or missing
        # When tool calls are present, content might be None
        content = _dict.get("content") or ""

        additional_kwargs = {}
        if _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(_dict["function_call"])

        tool_calls = []
        if _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = _dict["tool_calls"]
            for tool_call in _dict["tool_calls"]:
                try:
                    tool_calls.append(
                        ToolCall(
                            name=tool_call["function"]["name"],
                            args=json.loads(tool_call["function"]["arguments"]),
                            id=tool_call.get("id"),
                        )
                    )
                except (KeyError, json.JSONDecodeError) as e:
                    logger.debug(f"Skipping malformed tool call: {e}")
                    continue

        return AIMessage(
            content=content, additional_kwargs=additional_kwargs, tool_calls=tool_calls
        )
    if role == "system":
        return SystemMessage(content=_dict["content"])
    if role == "function":
        return FunctionMessage(content=_dict["content"], name=_dict["name"])
    if role == "tool":
        return ToolMessage(content=_dict["content"], tool_call_id=_dict["tool_call_id"])
    if not role:
        return ChatMessage(content=_dict["content"], role="unknown")
    return ChatMessage(content=_dict["content"], role=role)


def _convert_delta_to_message_chunk(
    delta: Any, default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    """Convert a streaming delta to a LangChain message chunk.

    Args:
        delta: Delta object from streaming response
        default_class: Default message chunk class to use

    Returns:
        Appropriate BaseMessageChunk subclass
    """
    # Handle both dict and object-style deltadi
    if isinstance(delta, dict):
        role = delta.get("role")
        content = delta.get("content") or ""
        function_call = delta.get("function_call")
        raw_tool_calls = delta.get("tool_calls")
        reasoning = delta.get("reasoning")
    else:
        role = getattr(delta, "role", None)
        content = getattr(delta, "content", None) or ""
        function_call = getattr(delta, "function_call", None)
        raw_tool_calls = getattr(delta, "tool_calls", None)
        reasoning = getattr(delta, "reasoning", None)

    additional_kwargs: dict[str, Any] = {}
    if function_call:
        additional_kwargs["function_call"] = dict(function_call)

    if reasoning:
        reasoning_content = (
            reasoning.content
            if hasattr(reasoning, "content")
            else reasoning.get("content")
        )
        if reasoning_content:
            additional_kwargs["reasoning_content"] = reasoning_content

    tool_call_chunks = []
    if raw_tool_calls:
        additional_kwargs["tool_calls"] = raw_tool_calls
        for rtc in raw_tool_calls:
            try:
                if isinstance(rtc, dict):
                    func = rtc.get("function")
                    if func and isinstance(func, dict):
                        tool_call_chunks.append(
                            ToolCallChunk(
                                name=func.get("name") or "",
                                args=func.get("arguments") or "",
                                id=rtc.get("id"),
                                index=rtc.get("index"),
                            )
                        )
                else:
                    # Handle object-style tool call
                    func = getattr(rtc, "function", None)
                    if func:
                        tool_call_chunks.append(
                            ToolCallChunk(
                                name=getattr(func, "name", None) or "",
                                args=getattr(func, "arguments", None) or "",
                                id=getattr(rtc, "id", None),
                                index=getattr(rtc, "index", None),
                            )
                        )
            except (KeyError, AttributeError, TypeError) as e:
                # Log the error but continue processing other tool calls
                # This handles malformed tool call chunks in streaming
                logger.debug(f"Skipping malformed tool call chunk: {e}")
                continue

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    if role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_call_chunks=tool_call_chunks,
        )
    if role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    if default_class == FunctionMessageChunk:
        if function_call:
            if isinstance(function_call, dict):
                return FunctionMessageChunk(
                    content=function_call.get("arguments", ""),
                    name=function_call.get("name", ""),
                )
            else:
                return FunctionMessageChunk(
                    content=function_call.arguments or "",
                    name=function_call.name or "",
                )
    if role == "tool" or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)  # type: ignore[arg-type]
    return default_class(content=content)  # type: ignore[call-arg]


def _lc_tool_call_to_openai_tool_call(tool_call: ToolCall) -> dict[str, Any]:
    """Convert a LangChain ToolCall to OpenAI tool call format.

    Args:
        tool_call: LangChain ToolCall object

    Returns:
        Dictionary in OpenAI tool call format
    """
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"]),
        },
    }


def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    """Convert a LangChain message to a dictionary.

    Args:
        message: LangChain message object

    Returns:
        Dictionary representation of the message

    Raises:
        ValueError: If message type is unknown
    """
    message_dict: dict[str, Any] = {"content": message.content}
    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
        if message.tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_openai_tool_call(tc) for tc in message.tool_calls
            ]
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    elif isinstance(message, FunctionMessage):
        message_dict["role"] = "function"
        message_dict["name"] = message.name
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
        message_dict["tool_call_id"] = message.tool_call_id
    else:
        error_message = f"Got unknown type {message}"
        raise ValueError(error_message)
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict
