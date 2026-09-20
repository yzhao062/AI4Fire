"""Gateway helpers: read the key from the environment, as the gateway README documents, and call the chat endpoint."""
import ast
import json
import os
import re
import httpx

BASE = os.environ.get("NAIRR_GATEWAY_URL")


def get_base():
    base = os.environ.get("NAIRR_GATEWAY_URL") or BASE
    if not base:
        raise SystemExit("set NAIRR_GATEWAY_URL first; gateway endpoint is required for model calls; see README.md")
    return base.rstrip("/")


def load_key():
    """Return the gateway key from NAIRR_GATEWAY_KEY. The key never appears in this repository or in any log."""
    key = os.environ.get("NAIRR_GATEWAY_KEY")
    if not key:
        raise SystemExit("set NAIRR_GATEWAY_KEY first; see README.md")
    return key


def models(key):
    r = httpx.get(get_base() + "/models", headers={"Authorization": "Bearer " + key}, timeout=30)
    r.raise_for_status()
    return sorted(d["id"] for d in r.json()["data"])


def _bedrock_blocks(content):
    """Turn OpenAI-style message content (a string or a list of text and image_url parts) into converse blocks.

    The runners send images as base64 data URLs; Bedrock wants the raw bytes and the format name.
    """
    import base64

    if isinstance(content, str):
        return [{"text": content}]
    blocks = []
    for part in content:
        if part.get("type") == "text":
            blocks.append({"text": part["text"]})
        elif part.get("type") == "image_url":
            url = part["image_url"]["url"]
            head, data = url.split(",", 1)
            fmt = head.split("/", 1)[1].split(";", 1)[0]
            blocks.append({"image": {"format": "jpeg" if fmt == "jpg" else fmt, "source": {"bytes": base64.b64decode(data)}}})
        else:
            raise ValueError("unsupported content part: %r" % part.get("type"))
    return blocks


def bedrock_chat(model, messages, temperature=0, max_tokens=512):
    """Call a model on Bedrock. Credentials come from AWS_BEARER_TOKEN_BEDROCK or the standard chain.

    Reasoning models put a reasoningContent block before the text, so every text block is collected.
    Throttling is retried by botocore's adaptive mode rather than surfacing as an item error.
    """
    import boto3
    from botocore.config import Config

    client = boto3.client("bedrock-runtime", region_name="us-east-1",
                          config=Config(retries={"max_attempts": 10, "mode": "adaptive"}, read_timeout=600))
    system = [{"text": m["content"]} for m in messages if m["role"] == "system"]
    turns = [{"role": m["role"], "content": _bedrock_blocks(m["content"])} for m in messages if m["role"] != "system"]
    kwargs = {"modelId": model, "messages": turns,
              "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature}}
    if system:
        kwargs["system"] = system
    resp = client.converse(**kwargs)
    parts = resp["output"]["message"]["content"]
    text = "\n".join(p["text"] for p in parts if "text" in p)
    usage = resp.get("usage", {})
    return text, {"prompt_tokens": usage.get("inputTokens", 0), "completion_tokens": usage.get("outputTokens", 0),
                  "stop_reason": resp.get("stopReason")}, model


def call(key, model, messages, **kw):
    """Route by model name: a Bedrock id carries a dot before its first slash-free segment, a gateway name does not."""
    if model.startswith("bedrock:"):
        return bedrock_chat(model.split(":", 1)[1], messages, **kw)
    return chat(key, model, messages, **kw)


# The OpenAI reasoning models behind the gateway (gpt-5.x, gpt-6-astra, served through Azure) reject
# max_tokens in favour of max_completion_tokens, and reject every temperature except their default of 1.
# Probed on 2026-09-16: both rejections are HTTP 400 with an explicit message. The cap therefore counts
# reasoning tokens for these models, which the stored usage records under completion_tokens_details.
OPENAI_REASONING_PREFIXES = ("gpt-",)


def chat(key, model, messages, temperature=0, max_tokens=512, timeout=180):
    body = {"model": model, "messages": messages}
    if model.startswith(OPENAI_REASONING_PREFIXES):
        body["max_completion_tokens"] = max_tokens
    else:
        body["temperature"] = temperature
        body["max_tokens"] = max_tokens
    r = httpx.post(get_base() + "/chat/completions", headers={"Authorization": "Bearer " + key}, json=body, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    return d["choices"][0]["message"]["content"], d.get("usage", {}), d.get("model", model)


GEMMA_TOOL_PREFIXES = ("google.gemma-",)


def is_gemma_model(model: str) -> bool:
    """Return True if model uses Gemma's tool_code convention."""
    return model.startswith(GEMMA_TOOL_PREFIXES)


def render_gemma_tool_specs(tools) -> str:
    """Render tool specifications and the tool_code / tool_output protocol into system text."""
    specs = []
    for t in tools:
        if "toolSpec" in t:
            ts = t["toolSpec"]
            name = ts.get("name", "")
            desc = ts.get("description", "")
            schema = ts.get("inputSchema", {}).get("json", ts.get("inputSchema", {}))
            props = schema.get("properties", {})
        elif "function" in t:
            fn = t["function"]
            name = fn.get("name", "")
            desc = fn.get("description", "")
            schema = fn.get("parameters", {})
            props = schema.get("properties", {})
        else:
            name = t.get("name", "")
            desc = t.get("description", "")
            props = t.get("parameters", {}).get("properties", {})
        spec = f"- Tool: {name}\n  Description: {desc}\n  Parameters: {json.dumps(props)}"
        specs.append(spec)

    tools_text = "\n".join(specs)
    return (
        f"Available tools:\n{tools_text}\n\n"
        "To call a tool, output a Python call in a ```tool_code``` block. "
        "The response will appear in a ```tool_output``` block."
    )


def render_gemma_tool_result(content: str) -> str:
    """Format tool result string into a tool_output fence."""
    return f"```tool_output\n{content}\n```"


def parse_tool_code_fences(text: str, tools=None):
    """Parse tool_code fenced blocks into OpenAI-style tool_calls."""
    known_tools = {}
    if tools:
        for t in tools:
            if "toolSpec" in t:
                ts = t["toolSpec"]
                name = ts.get("name")
                schema = ts.get("inputSchema", {}).get("json", ts.get("inputSchema", {}))
                props = list(schema.get("properties", {}).keys())
                known_tools[name] = props
            elif "function" in t:
                fn = t["function"]
                name = fn.get("name")
                schema = fn.get("parameters", {})
                props = list(schema.get("properties", {}).keys())
                known_tools[name] = props

    pattern = re.compile(r"```tool_code\s*([\s\S]*?)```")
    matches = pattern.findall(text)
    tool_calls = []

    for idx, raw_code in enumerate(matches):
        code = raw_code.strip()
        try:
            tree = ast.parse(code)
        except Exception:
            continue

        for stmt in tree.body:
            if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
                continue
            call = stmt.value
            if isinstance(call.func, ast.Name) and call.func.id == "print" and call.args:
                if isinstance(call.args[0], ast.Call):
                    call = call.args[0]

            if not isinstance(call.func, ast.Name):
                continue
            func_name = call.func.id

            if tools is not None and func_name not in known_tools:
                continue

            param_names = known_tools.get(func_name, [])
            args_dict = {}

            failed = False
            for p_idx, arg_node in enumerate(call.args):
                try:
                    val = ast.literal_eval(arg_node)
                except Exception:
                    try:
                        val = ast.unparse(arg_node)
                    except Exception:
                        failed = True
                        break
                if p_idx < len(param_names):
                    args_dict[param_names[p_idx]] = val
                elif len(call.args) == 1 and param_names:
                    args_dict[param_names[0]] = val
                else:
                    args_dict[f"arg_{p_idx}"] = val

            if failed:
                continue

            for kw in call.keywords:
                if not kw.arg:
                    continue
                try:
                    val = ast.literal_eval(kw.value)
                except Exception:
                    try:
                        val = ast.unparse(kw.value)
                    except Exception:
                        failed = True
                        break
                args_dict[kw.arg] = val

            if failed:
                continue

            call_id = f"call_{func_name}_{idx}"
            tool_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(args_dict)
                }
            })

    return tool_calls


def _tool_blocks_as_text(turns):
    """Rewrite toolUse and toolResult blocks into text blocks, merging consecutive same-role turns.

    Used only for a toolless final call after a tool exchange; a run that never exhausts its tool budget
    never reaches it.
    """
    import json

    out = []
    for turn in turns:
        blocks = []
        for b in turn["content"]:
            if "toolUse" in b:
                tu = b["toolUse"]
                blocks.append({"text": "[Tool call %s(%s)]" % (tu.get("name", ""), json.dumps(tu.get("input", {})))})
            elif "toolResult" in b:
                texts = [c.get("text", "") for c in b["toolResult"].get("content", []) if "text" in c]
                blocks.append({"text": "[Tool result]\n" + "\n".join(texts)})
            else:
                blocks.append(b)
        if out and out[-1]["role"] == turn["role"]:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": turn["role"], "content": blocks})
    return out


def bedrock_call_tools(model, messages, tools=None, temperature=0, max_tokens=512):
    """Call a model on Bedrock with tools using the Converse API or Gemma tool_code convention."""
    import json
    import boto3
    from botocore.config import Config

    client = boto3.client("bedrock-runtime", region_name="us-east-1",
                          config=Config(retries={"max_attempts": 10, "mode": "adaptive"}, read_timeout=600))
    is_gemma = is_gemma_model(model)
    system = [{"text": m["content"]} for m in messages if m["role"] == "system"]

    if is_gemma and tools:
        tool_spec_text = render_gemma_tool_specs(tools)
        if system:
            system = [{"text": system[0]["text"] + "\n\n" + tool_spec_text}]
        else:
            system = [{"text": tool_spec_text}]

    turns = []
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "tool":
            if is_gemma:
                tool_res_block = {"text": render_gemma_tool_result(str(m.get("content", "")))}
            else:
                tool_res_block = {
                    "toolResult": {
                        "toolUseId": m.get("tool_call_id", ""),
                        "content": [{"text": str(m.get("content", ""))}],
                        "status": "success" if not m.get("is_error") else "error"
                    }
                }
            if turns and turns[-1]["role"] == "user":
                turns[-1]["content"].append(tool_res_block)
            else:
                turns.append({"role": "user", "content": [tool_res_block]})
        elif m["role"] == "assistant":
            blocks = []
            if m.get("content"):
                if isinstance(m["content"], str):
                    blocks.append({"text": m["content"]})
                elif isinstance(m["content"], list):
                    blocks.extend(m["content"])
            if not is_gemma and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {"raw": args}
                    blocks.append({
                        "toolUse": {
                            "toolUseId": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": args
                        }
                    })
            turns.append({"role": "assistant", "content": blocks})
        else:
            if isinstance(m.get("content"), list):
                turns.append({"role": m["role"], "content": m["content"]})
            else:
                turns.append({"role": m["role"], "content": _bedrock_blocks(m["content"])})

    if not tools and not is_gemma:
        # A final call without tools (the runner's eight-call cap) cannot carry toolUse or toolResult blocks:
        # Bedrock rejects them without a toolConfig ("The toolConfig field must be defined when using toolUse
        # and toolResult content blocks"). Render the tool exchange as text so the model still sees it.
        turns = _tool_blocks_as_text(turns)

    kwargs = {
        "modelId": model,
        "messages": turns,
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature}
    }
    if system:
        kwargs["system"] = system

    if tools and not is_gemma:
        specs = []
        for t in tools:
            if "toolSpec" in t:
                specs.append(t)
            elif "function" in t:
                fn = t["function"]
                specs.append({
                    "toolSpec": {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "inputSchema": {"json": fn.get("parameters", {})}
                    }
                })
            else:
                specs.append(t)
        kwargs["toolConfig"] = {"tools": specs}

    resp = client.converse(**kwargs)
    out_msg = resp["output"]["message"]
    parts = out_msg.get("content", [])
    text = "\n".join(p["text"] for p in parts if "text" in p)

    tool_calls = []
    if is_gemma:
        tool_calls = parse_tool_code_fences(text, tools=tools)
    else:
        for p in parts:
            if "toolUse" in p:
                tu = p["toolUse"]
                tool_calls.append({
                    "id": tu["toolUseId"],
                    "type": "function",
                    "function": {
                        "name": tu["name"],
                        "arguments": json.dumps(tu.get("input", {}))
                    }
                })

    msg = {"role": "assistant", "content": text}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if is_gemma:
        msg["tool_protocol"] = "tool_code"
    msg["bedrock_content"] = parts

    usage = {
        "prompt_tokens": resp.get("usage", {}).get("inputTokens", 0),
        "completion_tokens": resp.get("usage", {}).get("outputTokens", 0),
        "stop_reason": resp.get("stopReason")
    }
    if is_gemma:
        usage["tool_protocol"] = "tool_code"
    return msg, usage, model


def gateway_call_tools(key, model, messages, tools=None, temperature=0, max_tokens=512, timeout=180):
    """Call a model on the OpenAI-compatible gateway with tools."""
    body = {"model": model, "messages": messages}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if model.startswith(OPENAI_REASONING_PREFIXES):
        body["max_completion_tokens"] = max_tokens
    else:
        body["temperature"] = temperature
        body["max_tokens"] = max_tokens
    r = httpx.post(get_base() + "/chat/completions", headers={"Authorization": "Bearer " + key}, json=body, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    msg = d["choices"][0]["message"]
    return msg, d.get("usage", {}), d.get("model", model)


# The Azure endpoint behind gpt-6-astra rejects function tools on /chat/completions unless reasoning is switched
# off ("Function tools with reasoning_effort are not supported ... use /v1/responses or set reasoning_effort to
# 'none'", HTTP 400, probed 2026-09-17). The OpenAI reasoning models therefore take the Responses API for tool
# calls, with reasoning left at the endpoint's default so the arm stays comparable to the bare arm. The chat-style
# message list the runner keeps is converted on the way in, and the raw output items of each response are carried
# back verbatim on the assistant message (under _responses_output) so that the reasoning items travel with their
# function calls on the next turn, which the API requires.
def _to_responses_input(messages):
    items = []
    for m in messages:
        if m.get("_responses_output") is not None:
            items.extend(m["_responses_output"])
        elif m.get("role") == "tool":
            items.append({"type": "function_call_output", "call_id": m.get("tool_call_id", ""), "output": m.get("content", "")})
        elif m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments", "")
                items.append({"type": "function_call", "call_id": tc.get("id", ""), "name": fn.get("name", ""),
                              "arguments": args if isinstance(args, str) else json.dumps(args)})
            if m.get("content"):
                items.append({"role": "assistant", "content": m["content"]})
        else:
            items.append({"role": m["role"], "content": m["content"]})
    return items


def gateway_call_tools_responses(key, model, messages, tools=None, max_tokens=512, timeout=180, **kw):
    """Call an OpenAI reasoning model on the gateway through the Responses API; return a chat-shaped message."""
    body = {"model": model, "input": _to_responses_input(messages), "max_output_tokens": max_tokens}
    if tools:
        body["tools"] = [{"type": "function", "name": t["function"]["name"], "description": t["function"].get("description", ""),
                          "parameters": t["function"].get("parameters", {})} for t in tools]
        body["tool_choice"] = "auto"
    r = httpx.post(get_base() + "/responses", headers={"Authorization": "Bearer " + key}, json=body, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    out = d.get("output", []) or []
    text = "".join(c.get("text", "") for o in out if o.get("type") == "message"
                   for c in (o.get("content") or []) if c.get("type") == "output_text")
    tool_calls = [{"id": o.get("call_id", ""), "type": "function",
                   "function": {"name": o.get("name", ""), "arguments": o.get("arguments", "")}}
                  for o in out if o.get("type") == "function_call"]
    msg = {"role": "assistant", "content": text or None, "_responses_output": out}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    u = d.get("usage", {}) or {}
    usage = {"prompt_tokens": u.get("input_tokens", 0) or 0, "completion_tokens": u.get("output_tokens", 0) or 0}
    # A detail the endpoint did not report stays absent, so a stored row can tell "zero" from "not measured"
    # (the 2026-09-17 tool-arm rows carry no reasoning-token field for that reason; see the paper's D.18).
    otd = u.get("output_tokens_details") or {}
    if "reasoning_tokens" in otd:
        usage["completion_tokens_details"] = {"reasoning_tokens": otd.get("reasoning_tokens") or 0}
    itd = u.get("input_tokens_details") or {}
    if "cached_tokens" in itd:
        usage["prompt_tokens_details"] = {"cached_tokens": itd.get("cached_tokens") or 0}
    return msg, usage, d.get("model", model)


def call_tools(key, model, messages, tools=None, max_tokens=512, **kw):
    """Route by model name: a Bedrock id starts with 'bedrock:', an OpenAI reasoning model takes the Responses API,
    and every other gateway name takes chat completions."""
    if model.startswith("bedrock:"):
        return bedrock_call_tools(model.split(":", 1)[1], messages, tools=tools, max_tokens=max_tokens, **kw)
    if model.startswith(OPENAI_REASONING_PREFIXES):
        return gateway_call_tools_responses(key, model, messages, tools=tools, max_tokens=max_tokens, **kw)
    return gateway_call_tools(key, model, messages, tools=tools, max_tokens=max_tokens, **kw)


if __name__ == "__main__":
    k = load_key()
    print("gateway models:", ", ".join(models(k)))
    text, usage, served = chat(k, "claude-opus-5", [{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=16)
    print("claude-opus-5 ->", repr(text.strip()), "| served:", served, "| usage:", usage)
