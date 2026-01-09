"""Basic usage examples for langchain-anyllm.

Note: You need to have the appropriate API key available for your chosen provider. API keys can be passed explicitly via the `api_key` parameter, or set as environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.). See the [any-llm documentation](https://mozilla-ai.github.io/any-llm/providers/) for provider-specific requirements.

    python examples/basic_usage.py

Or run individual async examples:
    python -c "import asyncio; from basic_usage import basic_chat; asyncio.run(basic_chat())"
"""

import asyncio
from langchain_anyllm import ChatAnyLLM


def basic_chat_sync() -> None:
    """Synchronous basic chat example."""
    llm = ChatAnyLLM(model="openai:gpt-4", model_kwargs={"temperature": 0.7})

    messages = [
        ("system", "You are a helpful assistant."),
        ("human", "What is the capital of France?"),
    ]

    response = llm.invoke(messages)
    print(f"Response: {response.content}")


def streaming_example_sync() -> None:
    """Synchronous streaming response example."""
    llm = ChatAnyLLM(model="openai:gpt-4", model_kwargs={"temperature": 0.7})

    messages = [("human", "Tell me a short story about a robot.")]

    print("Streaming response: ", end="")
    for chunk in llm.stream(messages):
        print(chunk.content, end="", flush=True)
    print()


def tool_calling_example_sync() -> None:
    """Synchronous tool calling example."""
    from pydantic import BaseModel, Field

    class GetWeather(BaseModel):
        """Get the current weather in a given location."""

        location: str = Field(..., description="City and state, e.g. San Francisco, CA")
        unit: str = Field(
            "celsius", description="Temperature unit (celsius or fahrenheit)"
        )

    llm = ChatAnyLLM(model="openai:gpt-4", model_kwargs={"temperature": 0})
    llm_with_tools = llm.bind_tools([GetWeather])

    response = llm_with_tools.invoke("What's the weather like in Paris?")

    # Pretty print the tool calls
    if response.tool_calls:
        print(f"The model wants to call {len(response.tool_calls)} tool(s):")
        for i, tool_call in enumerate(response.tool_calls, 1):
            print(f"  {i}. Tool: {tool_call['name']}")
            print(f"     Arguments: {tool_call['args']}")
    else:
        print(f"No tool calls. Response: {response.content}")


def structured_output_example_sync() -> None:
    """Synchronous structured output example using tool calling."""
    from pydantic import BaseModel, Field

    class Person(BaseModel):
        """Information about a person."""

        name: str = Field(..., description="The person's name")
        age: int = Field(..., description="The person's age")
        occupation: str = Field(..., description="The person's occupation")

    llm = ChatAnyLLM(model="openai:gpt-4", model_kwargs={"temperature": 0})

    # Use bind_tools instead of with_structured_output to avoid tool_choice issues
    llm_with_tools = llm.bind_tools([Person])

    response = llm_with_tools.invoke(
        "Extract information: John is a 30-year-old software engineer."
    )

    # Extract the structured data from tool calls
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        print(f"Structured output: {tool_call['args']}")
    else:
        print(f"Response: {response.content}")


async def basic_chat() -> None:
    """Async basic chat example."""
    llm = ChatAnyLLM(model="openai:gpt-4", model_kwargs={"temperature": 0.7})

    messages = [
        ("system", "You are a helpful assistant."),
        ("human", "What is the capital of France?"),
    ]

    response = await llm.ainvoke(messages)
    print(f"Response: {response.content}")


async def streaming_example() -> None:
    """Async streaming response example."""
    llm = ChatAnyLLM(model="openai:gpt-4", model_kwargs={"temperature": 0.7})

    messages = [("human", "Tell me a short story about a robot.")]

    print("Streaming response: ", end="")
    async for chunk in llm.astream(messages):
        print(chunk.content, end="", flush=True)
    print()


async def tool_calling_example() -> None:
    """Async tool calling example."""
    from pydantic import BaseModel, Field

    class GetWeather(BaseModel):
        """Get the current weather in a given location."""

        location: str = Field(..., description="City and state, e.g. San Francisco, CA")
        unit: str = Field(
            "celsius", description="Temperature unit (celsius or fahrenheit)"
        )

    llm = ChatAnyLLM(model="openai:gpt-4", model_kwargs={"temperature": 0})
    llm_with_tools = llm.bind_tools([GetWeather])

    response = await llm_with_tools.ainvoke("What's the weather like in Paris?")

    # Pretty print the tool calls
    if response.tool_calls:
        print(f"The model wants to call {len(response.tool_calls)} tool(s):")
        for i, tool_call in enumerate(response.tool_calls, 1):
            print(f"  {i}. Tool: {tool_call['name']}")
            print(f"     Arguments: {tool_call['args']}")
    else:
        print(f"No tool calls. Response: {response.content}")


async def structured_output_example() -> None:
    """Async structured output example using tool calling."""
    from pydantic import BaseModel, Field

    class Person(BaseModel):
        """Information about a person."""

        name: str = Field(..., description="The person's name")
        age: int = Field(..., description="The person's age")
        occupation: str = Field(..., description="The person's occupation")

    llm = ChatAnyLLM(model="openai:gpt-4", model_kwargs={"temperature": 0})

    # Use bind_tools instead of with_structured_output to avoid tool_choice issues
    llm_with_tools = llm.bind_tools([Person])

    response = await llm_with_tools.ainvoke(
        "Extract information: John is a 30-year-old software engineer."
    )

    # Extract the structured data from tool calls
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        print(f"Structured output: {tool_call['args']}")
    else:
        print(f"Response: {response.content}")


def run_sync_examples() -> None:
    """Run all synchronous examples."""
    print("=== Basic Chat (Sync) ===")
    try:
        basic_chat_sync()
    except Exception as e:
        print(f"Error in basic_chat_sync: {e}")

    print("\n=== Streaming (Sync) ===")
    try:
        streaming_example_sync()
    except Exception as e:
        print(f"Error in streaming_example_sync: {e}")

    print("\n=== Tool Calling (Sync) ===")
    try:
        tool_calling_example_sync()
    except Exception as e:
        print(f"Error in tool_calling_example_sync: {e}")

    print("\n=== Structured Output (Sync) ===")
    try:
        structured_output_example_sync()
    except Exception as e:
        print(f"Error in structured_output_example_sync: {e}")


async def run_async_examples() -> None:
    """Run all asynchronous examples."""
    print("\n=== Basic Chat (Async) ===")
    try:
        await basic_chat()
    except Exception as e:
        print(f"Error in basic_chat: {e}")

    print("\n=== Streaming (Async) ===")
    try:
        await streaming_example()
    except Exception as e:
        print(f"Error in streaming_example: {e}")

    print("\n=== Tool Calling (Async) ===")
    try:
        await tool_calling_example()
    except Exception as e:
        print(f"Error in tool_calling_example: {e}")

    print("\n=== Structured Output (Async) ===")
    try:
        await structured_output_example()
    except Exception as e:
        print(f"Error in structured_output_example: {e}")


async def main() -> None:
    """Run all examples."""
    # Run sync examples first
    run_sync_examples()

    # Then run async examples
    await run_async_examples()


if __name__ == "__main__":
    asyncio.run(main())
