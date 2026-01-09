# Debugging Integration Test Failures

## Quick Diagnosis

Run the debug script to test basic functionality:

```bash
export OPENAI_API_KEY=your-key-here
uv run python test_debug.py
```

This will test:
1. Basic invoke (checks if content is properly returned)
2. Tool calling (checks tool_calls handling)
3. Streaming (checks streaming chunks)

## Known Issues Fixed

### 1. ✅ tool_choice Translation
- **Fixed**: `tool_choice='any'` now translates to `'required'` for OpenAI
- **Fixed**: `tool_choice` only included when `tools` are present
- **Location**: `langchain_anyllm/chat_models.py:121-140`

### 2. ✅ Anthropic Format Errors
- **Fixed**: Set `supports_anthropic_inputs = False` in test config
- **Reason**: Using OpenAI model which doesn't support Anthropic `content_blocks` or `tool_use` formats
- **Location**: `tests/integration_tests/test_standard.py:58-65`

## Remaining Error Categories (25 failures)

### Category 1: Assertion Failures (10+ tests)
**Examples:**
- `test_invoke` - AssertionError: assert False
- `test_conversation` - assert False
- `test_batch` - AssertionError: assert False

**Possible Causes:**
1. Empty content being returned
2. Content not being a string when expected
3. Missing required message attributes

**Debug Steps:**
```python
# Check what the actual response looks like
result = llm.invoke("Hello")
print(f"Content: {result.content!r}")
print(f"Type: {type(result.content)}")
print(f"Length: {len(result.content) if result.content else 0}")
```

### Category 2: RuntimeError: Event loop is closed (6 tests)
**Examples:**
- `test_stream[model0]`
- `test_usage_metadata_streaming`
- `test_tool_calling_with_no_arguments`

**Possible Causes:**
1. Async event loop lifecycle issues
2. pytest-asyncio configuration problem
3. Generator not being properly closed

**Potential Fix:**
Check if we need to add `asyncio_default_fixture_loop_scope` to pytest config.

### Category 3: TypeError: issubclass() arg 1 must be a class (3 tests)
**Examples:**
- `test_structured_output[typeddict]`
- `test_structured_output_async[typeddict]`
- `test_structured_output_optional_param`

**Possible Cause:**
TypedDict is not being handled correctly in `with_structured_output()`.

**Debug:**
Check if `with_structured_output()` from BaseChatModel properly handles TypedDict vs Pydantic models.

### Category 4: Tool Choice Dictionary Format (1 test)
**Example:**
- `test_tool_choice` - Invalid value: 'magic_function'

**Error Message:**
```
openai.BadRequestError: Invalid value: 'magic_function'.
Supported values are: 'none', 'auto', and 'required'.
```

**Cause:**
The test passes a dictionary like:
```python
tool_choice={'type': 'function', 'function': {'name': 'magic_function'}}
```

But our code might be extracting the wrong value.

**Fix Needed:**
Check if we're properly passing through the dictionary format in `_create_params`.

### Category 5: Anthropic tool_use Format (2 tests)
**Examples:**
- `test_tool_message_histories_list_content`
- `test_anthropic_inputs`

**Error:**
```
Invalid value: 'tool_use'. Supported values are: 'text', 'image_url', ...
```

**Note:** These might still fail even with `supports_anthropic_inputs = False` if they're not properly skipped.

## Detailed Test Checklist

Run each category separately to isolate issues:

```bash
# Test basic invocation
pytest tests/integration_tests/test_standard.py::TestChatAnyLLMStandard::test_invoke -xvs

# Test streaming
pytest tests/integration_tests/test_standard.py::TestChatAnyLLMStandard::test_stream -xvs

# Test tool calling
pytest tests/integration_tests/test_standard.py::TestChatAnyLLMStandard::test_tool_calling -xvs

# Test structured output
pytest tests/integration_tests/test_standard.py::TestChatAnyLLMStandard::test_structured_output -xvs
```

## Expected Behavior

After fixes, the integration tests should:
- ✅ Pass: ~40-45 tests (basic invoke, stream, tools, structured output with Pydantic)
- ⚠️  Skip: 3-5 tests (features not supported like image/audio inputs)
- ❌ Fail: 0-3 tests (edge cases that may need provider-specific handling)

## Next Steps

1. Run `test_debug.py` to verify basic functionality works
2. Run one failing test at a time with `-xvs` to see full error
3. Check if errors are in our code or test assumptions
4. Update test configuration properties if needed
