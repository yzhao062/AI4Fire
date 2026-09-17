"""Gateway helpers: read the key from the environment, as the gateway README documents, and call the chat endpoint."""
import os

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


if __name__ == "__main__":
    k = load_key()
    print("gateway models:", ", ".join(models(k)))
    text, usage, served = chat(k, "claude-opus-5", [{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=16)
    print("claude-opus-5 ->", repr(text.strip()), "| served:", served, "| usage:", usage)
