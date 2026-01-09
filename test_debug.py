"""Debug script to test ChatAnyLLM with actual API calls.

Run with: OPENAI_API_KEY=your-key uv run python test_debug.py
"""

import os
from langchain_anyllm import ChatAnyLLM
from langchain_core.messages import HumanMessage

def test_basic_invoke():
    """Test basic invocation."""
    print("Testing basic invoke...")
    llm = ChatAnyLLM(
        model="openai:gpt-4o-mini",
        model_kwargs={"temperature": 0},
    )

    result = llm.invoke("Say 'Hello'")
    print(f"Result type: {type(result)}")
    print(f"Result content: {result.content}")
    print(f"Content length: {len(result.content)}")
    print(f"Content type: {type(result.content)}")
    assert result.content, "Content is empty!"
    assert isinstance(result.content, str), "Content is not a string!"
    print("✅ Basic invoke test passed\n")

def test_tool_calling():
    """Test tool calling."""
    print("Testing tool calling...")
    llm = ChatAnyLLM(
        model="openai:gpt-4o-mini",
        model_kwargs={"temperature": 0},
    )

    from langchain_core.tools import tool

    @tool
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"Weather in {city}: Sunny"

    llm_with_tools = llm.bind_tools([get_weather])
    result = llm_with_tools.invoke("What's the weather in Paris?")

    print(f"Result type: {type(result)}")
    print(f"Result content: {result.content}")
    print(f"Tool calls: {result.tool_calls}")
    print(f"Additional kwargs: {result.additional_kwargs}")
    print("✅ Tool calling test passed\n")

def test_streaming():
    """Test streaming."""
    print("Testing streaming...")
    llm = ChatAnyLLM(
        model="openai:gpt-4o-mini",
        model_kwargs={"temperature": 0},
    )

    chunks = []
    for chunk in llm.stream("Count to 3"):
        chunks.append(chunk)
        print(f"Chunk: {chunk.content}", end="", flush=True)

    print(f"\n\nTotal chunks: {len(chunks)}")
    full_content = "".join(str(chunk.content) for chunk in chunks)
    print(f"Full content: {full_content}")
    print("✅ Streaming test passed\n")

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Skipping tests.")
        exit(1)

    try:
        test_basic_invoke()
        test_tool_calling()
        test_streaming()
        print("\n✅ All debug tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
