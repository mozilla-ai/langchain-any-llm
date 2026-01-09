# langchain-anyllm

**One interface for every LLM.**

This integration enables you to use [any-llm's](https://github.com/mozilla-ai/any-llm) unified interface (supporting OpenAI, Anthropic, Gemini, local models, and more) as a standard LangChain `ChatModel`. See all `any-llm` supported providers [here](https://mozilla-ai.github.io/any-llm/providers/)

Stop rewriting your specific adapter code every time you want to test a new model. Switch between OpenAI, Anthropic, Gemini, and local models (via Ollama/LocalAI) just by changing a string.

## Installation

```bash
pip install langchain-anyllm
```

## Features

- **Unified Interface**: Use OpenAI, Anthropic, Google, or local models through a single API
- **Streaming Support**: Full support for both synchronous and asynchronous streaming
- **Tool Calling**: Native support for LangChain tool binding
- **Usage Tracking**: Automatic token usage metadata tracking
- **Multiple Providers**: See all supported providers [here](https://mozilla-ai.github.io/any-llm/providers/)

## Usage

**Note:** You need to have the appropriate API key available for your chosen provider. API keys can be passed explicitly via the `api_key` parameter, or set as environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.). See the [any-llm documentation](https://mozilla-ai.github.io/any-llm/providers/) for provider-specific requirements.

### Basic Chat

```python
from langchain_anyllm import ChatAnyLLM

# Initialize with any supported model
llm = ChatAnyLLM(model="openai:gpt-4", temperature=0.7)

# Invoke for a single response
response = llm.invoke("Tell me a joke")
print(response.content)
```

### Streaming

```python
from langchain_anyllm import ChatAnyLLM

llm = ChatAnyLLM(model="openai:gpt-4")

# Stream responses
for chunk in llm.stream("Write a poem about the ocean"):
    print(chunk.content, end="", flush=True)
```

### Async Support

```python
import asyncio
from langchain_anyllm import ChatAnyLLM

async def main():
    llm = ChatAnyLLM(model="openai:gpt-4")

    # Async invoke
    response = await llm.ainvoke("What is the meaning of life?")
    print(response.content)

    # Async streaming
    async for chunk in llm.astream("Count to 10"):
        print(chunk.content, end="", flush=True)

asyncio.run(main())
```

### Tool Calling

```python
from langchain_anyllm import ChatAnyLLM
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather for a location."""
    return f"The weather in {location} is sunny!"

llm = ChatAnyLLM(model="openai:gpt-4")
llm_with_tools = llm.bind_tools([get_weather])

response = llm_with_tools.invoke("What's the weather in San Francisco?")
print(response.tool_calls)
```

### Configuration

```python
from langchain_anyllm import ChatAnyLLM

llm = ChatAnyLLM(
    model="openai:gpt-4",
    api_key="your-api-key",  # Optional, reads from environment if not provided
    api_base="https://custom-endpoint.com/v1",  # Optional custom endpoint
    model_kwargs={
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 0.9,
    }
)
```

## Parameters

- `model` (str): The model to use (e.g., "openai:gpt-4", "anthropic:claude-3-sonnet-20240229")
- `api_key` (str, optional): API key for the provider. Reads from environment if not provided
- `api_base` (str, optional): Custom API endpoint
- `model_kwargs` (dict, optional): Additional parameters to pass to the model

## Supported Providers

any-llm supports a wide range of providers. See the [full list here](https://mozilla-ai.github.io/any-llm/providers/).

Common providers include:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Google (Gemini)
- Cohere
- Mistral
- Ollama (local models)
- And many more...

## Development

### Running Tests

```bash
uv run pytest tests/
```

### Type Checking

```bash
mypy langchain_anyllm/
```

### Linting

```bash
ruff check langchain_anyllm/
```

## License

MIT
