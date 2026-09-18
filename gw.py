"""Gateway helpers: read the key from the environment, as the gateway README documents, and call the chat endpoint."""
import os

import json
import httpx

BASE = "http://35.226.229.248:4000/v1"


def load_key():
    """Return the gateway key from NAIRR_GATEWAY_KEY. The key never appears in this repository or in any log."""
    key = os.environ.get("NAIRR_GATEWAY_KEY")
    if not key:
        raise SystemExit("set NAIRR_GATEWAY_KEY first; see ai-research-resources/nairr-pilot/README.md")
    return key


def models(key):
    r = httpx.get(BASE + "/models", headers={"Authorization": "Bearer " + key}, timeout=30)
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
    r = httpx.post(BASE + "/chat/completions", headers={"Authorization": "Bearer " + key}, json=body, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    return d["choices"][0]["message"]["content"], d.get("usage", {}), d.get("model", model)


def bedrock_call_tools(model, messages, tools=None, temperature=0, max_tokens=512):
    """Call a model on Bedrock with tools using the Converse API.

    Handles toolConfig with toolSpec, stopReason == 'tool_use', and toolUse blocks answered with toolResult blocks.
    """
    import json
    import boto3
    from botocore.config import Config

    client = boto3.client("bedrock-runtime", region_name="us-east-1",
                          config=Config(retries={"max_attempts": 10, "mode": "adaptive"}, read_timeout=600))
    system = [{"text": m["content"]} for m in messages if m["role"] == "system"]

    turns = []
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "tool":
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
            if m.get("tool_calls"):
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

    kwargs = {
        "modelId": model,
        "messages": turns,
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature}
    }
    if system:
        kwargs["system"] = system

    if tools:
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
    msg["bedrock_content"] = parts

    usage = {
        "prompt_tokens": resp.get("usage", {}).get("inputTokens", 0),
        "completion_tokens": resp.get("usage", {}).get("outputTokens", 0),
        "stop_reason": resp.get("stopReason")
    }
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
    r = httpx.post(BASE + "/chat/completions", headers={"Authorization": "Bearer " + key}, json=body, timeout=timeout)
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
    r = httpx.post(BASE + "/responses", headers={"Authorization": "Bearer " + key}, json=body, timeout=timeout)
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
    usage = {"prompt_tokens": u.get("input_tokens", 0) or 0, "completion_tokens": u.get("output_tokens", 0) or 0,
             "completion_tokens_details": {"reasoning_tokens": ((u.get("output_tokens_details") or {}).get("reasoning_tokens", 0) or 0)},
             "prompt_tokens_details": {"cached_tokens": ((u.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0)}}
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
