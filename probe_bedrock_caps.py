"""Probe every Bedrock chat model on the fire-bench key for the three capabilities the five tasks need:
plain text, image input (smoke detection and aerial question answering), and tool use (FPA-FOD tool use).

Each probe is one tiny Converse call, so the whole sweep costs cents. Results go to
probes/bedrock-caps-<date>.json and a table prints to stdout. The key comes from AWS_BEARER_TOKEN_BEDROCK
through boto3's standard chain and is never printed.

Usage: python probe_bedrock_caps.py [--models id1,id2,...] [--out probes/bedrock-caps-2026-09-18.json]
"""
import argparse
import base64
import datetime as dt
import io
import json
import pathlib
import time

import boto3
from botocore.config import Config

HANDOUT_MODELS = [
    # Amazon Nova
    "amazon.nova-micro-v1:0", "amazon.nova-lite-v1:0", "amazon.nova-pro-v1:0", "us.amazon.nova-2-lite-v1:0",
    # Meta Llama
    "us.meta.llama3-3-70b-instruct-v1:0", "us.meta.llama4-scout-17b-instruct-v1:0",
    "us.meta.llama4-maverick-17b-instruct-v1:0", "us.meta.llama3-1-70b-instruct-v1:0", "us.meta.llama3-1-8b-instruct-v1:0",
    # Mistral
    "mistral.mistral-large-3-675b-instruct", "mistral.mistral-small-2402-v1:0", "mistral.ministral-3-8b-instruct",
    "mistral.mixtral-8x7b-instruct-v0:1", "mistral.devstral-2-123b",
    # Qwen
    "qwen.qwen3-vl-235b-a22b", "qwen.qwen3-32b-v1:0", "qwen.qwen3-next-80b-a3b", "qwen.qwen3-coder-30b-a3b-v1:0",
    # DeepSeek
    "deepseek.v3.2", "us.deepseek.r1-v1:0",
    # OpenAI open weight
    "openai.gpt-oss-120b-1:0", "openai.gpt-oss-20b-1:0",
    # Google Gemma
    "google.gemma-3-27b-it", "google.gemma-3-12b-it", "google.gemma-3-4b-it",
    # Z.AI GLM
    "zai.glm-5", "zai.glm-4.7", "zai.glm-4.7-flash",
    # MiniMax, Moonshot, NVIDIA, Writer
    "minimax.minimax-m2.5", "moonshotai.kimi-k2.5", "moonshot.kimi-k2-thinking",
    "nvidia.nemotron-super-3-120b", "writer.palmyra-vision-7b",
]

TOOL = {"toolSpec": {"name": "get_number", "description": "Return the stored number for a name.",
                     "inputSchema": {"json": {"type": "object", "properties": {"name": {"type": "string"}},
                                              "required": ["name"]}}}}


def red_square_jpeg():
    """A 64x64 red JPEG, built without PIL if PIL is missing (PIL is present in py312, so this is the normal path)."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (220, 30, 30)).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def converse(client, model, content, tools=None, max_tokens=256):
    kwargs = {"modelId": model, "messages": [{"role": "user", "content": content}],
              "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0}}
    if tools:
        kwargs["toolConfig"] = {"tools": tools}
    t0 = time.time()
    resp = client.converse(**kwargs)
    parts = resp["output"]["message"]["content"]
    text = "\n".join(p["text"] for p in parts if "text" in p)
    tool_use = [p["toolUse"] for p in parts if "toolUse" in p]
    usage = resp.get("usage", {})
    return {"ok": True, "seconds": round(time.time() - t0, 1), "stop": resp.get("stopReason"),
            "text": text[:120], "tool_use": [{"name": t.get("name"), "input": t.get("input")} for t in tool_use],
            "in": usage.get("inputTokens"), "out": usage.get("outputTokens")}


def classify(exc):
    s = str(exc)
    for tag in ("AccessDeniedException", "ValidationException", "ThrottlingException", "ModelNotReadyException",
                "ResourceNotFoundException", "ModelErrorException", "ServiceUnavailableException"):
        if tag in s:
            return tag, s[:220]
    return type(exc).__name__, s[:220]


def probe(client, model, jpeg):
    out = {"model": model}
    for name, content, tools in (
        ("text", [{"text": "Reply with the single word OK."}], None),
        ("image", [{"image": {"format": "jpeg", "source": {"bytes": jpeg}}},
                   {"text": "What color is this image? Answer with one word."}], None),
        ("tool", [{"text": "Use the get_number tool to look up the value for the name alpha, then report it."}], [TOOL]),
    ):
        try:
            out[name] = converse(client, model, content, tools)
        except Exception as exc:  # every failure is data here
            tag, msg = classify(exc)
            out[name] = {"ok": False, "error": tag, "message": msg}
    r = out["tool"]
    out["tool_called"] = bool(r.get("ok") and r.get("tool_use"))
    out["image_ok"] = bool(out["image"].get("ok"))
    out["text_ok"] = bool(out["text"].get("ok"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma-separated model ids; default is the handout list")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    models = args.models.split(",") if args.models else HANDOUT_MODELS
    today = dt.date.today().isoformat()
    out_path = pathlib.Path(args.out or ("probes/bedrock-caps-%s.json" % today))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    client = boto3.client("bedrock-runtime", region_name="us-east-1",
                          config=Config(retries={"max_attempts": 4, "mode": "adaptive"}, read_timeout=180))
    jpeg = red_square_jpeg()
    results = []
    print("%-44s %-5s %-6s %-6s %s" % ("model", "text", "image", "tool", "note"))
    for m in models:
        r = probe(client, m, jpeg)
        results.append(r)
        note = ""
        for k in ("text", "image", "tool"):
            if not r[k].get("ok"):
                note = "%s: %s %s" % (k, r[k]["error"], r[k]["message"][:90])
                break
        if not note and r["tool"].get("ok") and not r["tool_called"]:
            note = "tool accepted but not called; text=%r" % r["tool"]["text"][:60]
        print("%-44s %-5s %-6s %-6s %s" % (m, "ok" if r["text_ok"] else "FAIL", "ok" if r["image_ok"] else "no",
                                           "ok" if r["tool_called"] else "no", note))
        out_path.write_text(json.dumps({"date": today, "region": "us-east-1", "results": results}, indent=1),
                            encoding="utf-8")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
