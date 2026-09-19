"""Unit tests for Gemma 3 tool calling through the tool_code convention."""
import json
import pathlib
from unittest.mock import MagicMock, patch

import pytest
import gw

S = pathlib.Path(__file__).parent.parent
PROBE_FILE = S / "probes" / "bedrock-caps-2026-09-18.json"

FPAFOD_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "query_fpafod",
            "description": "Execute a single SELECT query against the FPA-FOD SQLite database containing wildfire records in table 'Fires'. Returns a text table capped at 50 rows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL SELECT statement to execute."
                    }
                },
                "required": ["sql"]
            }
        }
    }
]

PROBE_TOOL = [
    {
        "toolSpec": {
            "name": "get_number",
            "description": "Return the stored number for a name.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "required": ["name"]
                }
            }
        }
    }
]

CALC_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Perform calculation with numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                    "ratio": {"type": "number"}
                },
                "required": ["count", "ratio"]
            }
        }
    }
]


def test_is_gemma_model():
    assert gw.is_gemma_model("google.gemma-3-27b-it") is True
    assert gw.is_gemma_model("google.gemma-3-12b-it") is True
    assert gw.is_gemma_model("google.gemma-3-4b-it") is True
    assert gw.is_gemma_model("amazon.nova-lite-v1:0") is False
    assert gw.is_gemma_model("us.meta.llama4-scout-17b-instruct-v1:0") is False


def test_system_text_rendering():
    """Verify tool name and the two fence names (tool_code, tool_output) appear once."""
    rendered = gw.render_gemma_tool_specs(FPAFOD_TOOL)
    assert rendered.count("query_fpafod") == 1
    assert rendered.count("tool_code") == 1
    assert rendered.count("tool_output") == 1

    rendered_probe = gw.render_gemma_tool_specs(PROBE_TOOL)
    assert rendered_probe.count("get_number") == 1
    assert rendered_probe.count("tool_code") == 1
    assert rendered_probe.count("tool_output") == 1


def test_result_block_rendering():
    """Verify tool result block rendering wraps content in tool_output fence."""
    res = gw.render_gemma_tool_result("STATE | count\nCA | 480")
    expected = "```tool_output\nSTATE | count\nCA | 480\n```"
    assert res == expected
    assert res.count("tool_output") == 1


def test_fence_parser_keyword_arguments():
    text = "```tool_code\nget_number(name='alpha')\n```"
    calls = gw.parse_tool_code_fences(text, tools=PROBE_TOOL)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_number"
    assert json.loads(calls[0]["function"]["arguments"]) == {"name": "alpha"}


def test_fence_parser_positional_arguments():
    text = "```tool_code\nquery_fpafod('SELECT count(*) FROM Fires WHERE STATE = \"CA\"')\n```"
    calls = gw.parse_tool_code_fences(text, tools=FPAFOD_TOOL)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "query_fpafod"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "sql": 'SELECT count(*) FROM Fires WHERE STATE = "CA"'
    }


def test_fence_parser_single_and_double_quotes():
    text_single = "```tool_code\nget_number(name='alpha')\n```"
    calls_single = gw.parse_tool_code_fences(text_single, tools=PROBE_TOOL)
    assert json.loads(calls_single[0]["function"]["arguments"]) == {"name": "alpha"}

    text_double = '```tool_code\nget_number(name="alpha")\n```'
    calls_double = gw.parse_tool_code_fences(text_double, tools=PROBE_TOOL)
    assert json.loads(calls_double[0]["function"]["arguments"]) == {"name": "alpha"}


def test_fence_parser_integers_and_floats():
    text = "```tool_code\ncalc(count=42, ratio=3.14)\n```"
    calls = gw.parse_tool_code_fences(text, tools=CALC_TOOL)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "calc"
    assert json.loads(calls[0]["function"]["arguments"]) == {"count": 42, "ratio": 3.14}

    text_pos = "```tool_code\ncalc(42, 3.14)\n```"
    calls_pos = gw.parse_tool_code_fences(text_pos, tools=CALC_TOOL)
    assert len(calls_pos) == 1
    assert calls_pos[0]["function"]["name"] == "calc"
    assert json.loads(calls_pos[0]["function"]["arguments"]) == {"count": 42, "ratio": 3.14}


def test_fence_parser_multiline_sql_string():
    sql = "SELECT COUNT(*)\nFROM Fires\nWHERE STATE = 'CA'\n  AND FIRE_YEAR = 2000"
    text = f'```tool_code\nquery_fpafod(sql="""{sql}""")\n```'
    calls = gw.parse_tool_code_fences(text, tools=FPAFOD_TOOL)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "query_fpafod"
    assert json.loads(calls[0]["function"]["arguments"]) == {"sql": sql}


def test_fence_parser_two_fences_in_one_answer():
    combined_tools = FPAFOD_TOOL + CALC_TOOL
    text = (
        "Let me run two operations.\n"
        "```tool_code\n"
        "query_fpafod('SELECT 1')\n"
        "```\n"
        "And calculate:\n"
        "```tool_code\n"
        "calc(count=10, ratio=0.5)\n"
        "```\n"
        "Awaiting results."
    )
    calls = gw.parse_tool_code_fences(text, tools=combined_tools)
    assert len(calls) == 2
    assert calls[0]["function"]["name"] == "query_fpafod"
    assert json.loads(calls[0]["function"]["arguments"]) == {"sql": "SELECT 1"}
    assert calls[1]["function"]["name"] == "calc"
    assert json.loads(calls[1]["function"]["arguments"]) == {"count": 10, "ratio": 0.5}


def test_fence_parser_unknown_tool():
    text = "```tool_code\nunknown_service(query='SELECT 1')\n```"
    calls = gw.parse_tool_code_fences(text, tools=FPAFOD_TOOL)
    assert calls == []


def test_fence_parser_malformed_fence():
    text = "```tool_code\ndef broken(\n```"
    calls = gw.parse_tool_code_fences(text, tools=FPAFOD_TOOL)
    assert calls == []


def test_probe_fixtures_from_json():
    """Verify probe recorded texts from probes/bedrock-caps-2026-09-18.json parse cleanly."""
    assert PROBE_FILE.exists(), f"probe file missing: {PROBE_FILE}"
    with open(PROBE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    gemma_entries = [r for r in data["results"] if r["model"].startswith("google.gemma-")]
    assert len(gemma_entries) >= 3

    for entry in gemma_entries:
        tool_text = entry["tool"]["text"]
        calls = gw.parse_tool_code_fences(tool_text, tools=PROBE_TOOL)
        assert len(calls) == 1, f"Failed parsing probe text for {entry['model']}: {tool_text}"
        assert calls[0]["function"]["name"] == "get_number"
        assert json.loads(calls[0]["function"]["arguments"]) == {"name": "alpha"}


def test_bedrock_call_tools_gemma_loop():
    """Mock boto3 client to test bedrock_call_tools request/response translation for Gemma."""
    mock_client = MagicMock()

    # Step 1: Model emits tool_code fence
    mock_client.converse.return_value = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "```tool_code\nquery_fpafod('SELECT 1')\n```"}]
            }
        },
        "usage": {"inputTokens": 150, "outputTokens": 20},
        "stopReason": "end_turn"
    }

    with patch("boto3.client", return_value=mock_client):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Count fires in CA."}
        ]
        msg, usage, model = gw.bedrock_call_tools(
            "google.gemma-3-27b-it", messages, tools=FPAFOD_TOOL
        )

    # Validate output shape
    assert msg["role"] == "assistant"
    assert msg.get("tool_protocol") == "tool_code"
    assert usage.get("tool_protocol") == "tool_code"
    assert len(msg.get("tool_calls", [])) == 1
    assert msg["tool_calls"][0]["function"]["name"] == "query_fpafod"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"sql": "SELECT 1"}

    # Validate converse call kwargs on way out
    _, kwargs = mock_client.converse.call_args
    assert "toolConfig" not in kwargs  # Gemma does not send toolConfig
    assert len(kwargs["system"]) == 1
    system_text = kwargs["system"][0]["text"]
    assert "You are a helpful assistant." in system_text
    assert "Available tools:" in system_text
    assert "query_fpafod" in system_text

    # Step 2: Next turn with tool result
    messages.append(msg)
    messages.append({
        "role": "tool",
        "tool_call_id": msg["tool_calls"][0]["id"],
        "content": "1"
    })

    # Model emits final answer without tool call
    mock_client.converse.return_value = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "The count is 1.\nANSWER: 1"}]
            }
        },
        "usage": {"inputTokens": 200, "outputTokens": 15},
        "stopReason": "end_turn"
    }

    with patch("boto3.client", return_value=mock_client):
        final_msg, final_usage, _ = gw.bedrock_call_tools(
            "google.gemma-3-27b-it", messages, tools=FPAFOD_TOOL
        )

    assert "tool_calls" not in final_msg
    assert final_msg["content"] == "The count is 1.\nANSWER: 1"
    assert final_msg.get("tool_protocol") == "tool_code"

    # Validate turns passed to converse on turn 2
    _, kwargs2 = mock_client.converse.call_args
    turns = kwargs2["messages"]
    assert len(turns) == 3
    assert turns[0]["role"] == "user"
    assert turns[1]["role"] == "assistant"
    assert turns[2]["role"] == "user"
    # User turn carries tool_output fence
    assert "```tool_output\n1\n```" in turns[2]["content"][0]["text"]


# The four shapes below are the forms the two smaller Gemma sizes actually returned on 2026-09-18,
# copied from task-tooluse/responses-bedrock_google.gemma-3-{12b,4b}-it-tool.jsonl.

RECORDED_ATTRIBUTE_CALL = """```tool_code
print(query_fpafod.query(sql="SELECT COUNT(*) FROM Fires WHERE STATE = 'AK' AND FIRE_YEAR = 1992"))
```"""

RECORDED_JSON_FENCE = """```tool_code
{"sql": "SELECT COUNT(*) FROM Fires WHERE STATE = 'AK' AND FIRE_YEAR = 1992"}
```"""

RECORDED_DICT_ARGUMENT = """```tool_code
query_fpafod(sql={"sql": "SELECT sum(FIRE_SIZE) FROM Fires WHERE STATE = 'MI' AND FIRE_YEAR = 2010"})
```"""

RECORDED_NAME_ARGUMENT = """```tool_code
print(query_fpafod(sql=sql))
```"""


def test_parser_on_recorded_gemma_failures():
    """The two zero scores of 2026-09-18 reproduce from the recorded text."""
    # Gemma 3 12B on 118 items: an undeclared attribute call is not a call of the declared tool
    assert gw.parse_tool_code_fences(RECORDED_ATTRIBUTE_CALL, tools=FPAFOD_TOOL) == []

    # Gemma 3 4B on all 156 items: a JSON object inside the fence is not a call
    assert gw.parse_tool_code_fences(RECORDED_JSON_FENCE, tools=FPAFOD_TOOL) == []

    # Gemma 3 12B on 3 items: a dictionary reaches the runner, which serializes it for the guard
    calls = gw.parse_tool_code_fences(RECORDED_DICT_ARGUMENT, tools=FPAFOD_TOOL)
    assert len(calls) == 1
    args = json.loads(calls[0]["function"]["arguments"])
    assert isinstance(args["sql"], dict)

    # Gemma 3 12B on 5 items: a bare name is not a literal, so the parser passes its text to the guard
    calls = gw.parse_tool_code_fences(RECORDED_NAME_ARGUMENT, tools=FPAFOD_TOOL)
    assert len(calls) == 1
    assert json.loads(calls[0]["function"]["arguments"]) == {"sql": "sql"}


def test_positional_and_print_forms_are_accepted():
    """The documented form is wider than keywords only, which the parser's callers rely on."""
    for text in ("```tool_code\nquery_fpafod('SELECT 1')\n```",
                 "```tool_code\nprint(query_fpafod(sql='SELECT 1'))\n```"):
        calls = gw.parse_tool_code_fences(text, tools=FPAFOD_TOOL)
        assert len(calls) == 1
        assert json.loads(calls[0]["function"]["arguments"]) == {"sql": "SELECT 1"}


def _tool_history():
    """A native (non-Gemma) exchange: question, tool call, tool result."""
    return [
        {"role": "user", "content": "How many fires burned in CA in 2010?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call-1", "function": {"name": "query_fpafod",
                                          "arguments": '{"sql": "SELECT count(*) FROM Fires"}'}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "count(*)\n8123"},
    ]


def _converse_kwargs(tools):
    """Run bedrock_call_tools against a mocked client and return the request it sent."""
    mock_client = MagicMock()
    mock_client.converse.return_value = {
        "output": {"message": {"role": "assistant", "content": [{"text": "ANSWER: 8123"}]}},
        "usage": {"inputTokens": 10, "outputTokens": 5},
        "stopReason": "end_turn",
    }
    with patch("boto3.client", return_value=mock_client):
        gw.bedrock_call_tools("us.amazon.nova-lite-v1:0", _tool_history(), tools=tools)
    _, kwargs = mock_client.converse.call_args
    return kwargs


def test_toolless_final_request_renders_tool_blocks_as_text():
    """After the eighth tool call the runner asks without tools; Bedrock rejects tool blocks there."""
    kwargs = _converse_kwargs(tools=None)
    blocks = [b for turn in kwargs["messages"] for b in turn["content"]]
    assert not any("toolUse" in b or "toolResult" in b for b in blocks)
    assert "toolConfig" not in kwargs
    text = "\n".join(b.get("text", "") for b in blocks)
    assert "[Tool call query_fpafod(" in text and "SELECT count(*) FROM Fires" in text
    assert "[Tool result]" in text and "8123" in text


def test_normal_request_keeps_native_tool_blocks():
    """The ordinary request still carries native toolUse and toolResult blocks."""
    kwargs = _converse_kwargs(tools=FPAFOD_TOOL)
    blocks = [b for turn in kwargs["messages"] for b in turn["content"]]
    assert any("toolUse" in b for b in blocks)
    assert any("toolResult" in b for b in blocks)
    assert "toolConfig" in kwargs
