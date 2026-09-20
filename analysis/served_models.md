# Served Model Identifiers and Date Ranges

This table records the exact serving path and model identifier returned by the model provider
gateway or API for each reported (task, model) pair, aggregated across bare and grounded conditions.

> [!NOTE]
> **Uniform Serving Confirmation**: Every benchmark run's rows carry exactly one `served_model` value.
> No run mixed multiple served model identifiers or provider endpoints.

| Task | Model (Paper Name) | Serving Path | Served Model Identifier (Call Count) | Total Calls | Date Range |
|---|---|:---:|---|:---:|:---:|
| Personnel allocation (ICS-209-PLUS) | **Llama 4 Maverick** (`bedrock_us.meta.llama4-maverick-17b-instruct-v1_0`) | Bedrock | `us.meta.llama4-maverick-17b-instruct-v1:0` (600) | 600 | 2026-09-16 |
| Personnel allocation (ICS-209-PLUS) | **Qwen3-VL-235B-A22B** (`bedrock_qwen.qwen3-vl-235b-a22b`) | Bedrock | `qwen.qwen3-vl-235b-a22b` (600) | 600 | 2026-09-16 |
| Personnel allocation (ICS-209-PLUS) | **claude-opus-4.8** | Gateway | `claude-opus-4.8` (600) | 600 | 2026-09-15 to 2026-09-16 |
| Personnel allocation (ICS-209-PLUS) | **claude-opus-5** | Gateway | `claude-opus-5` (600) | 600 | 2026-09-15 to 2026-09-16 |
| Personnel allocation (ICS-209-PLUS) | **gemini-3.1-pro** | Gateway | `gemini-3.1-pro` (600) | 600 | 2026-09-15 to 2026-09-16 |
| Personnel allocation (ICS-209-PLUS) | **gpt-6-astra** | Gateway | `gpt-6-astra` (600) | 600 | 2026-09-16 |
| Smoke detection (FIgLib) | **Llama 4 Maverick** (`bedrock_us.meta.llama4-maverick-17b-instruct-v1_0`) | Bedrock | `us.meta.llama4-maverick-17b-instruct-v1:0` (420) | 420 | 2026-09-16 |
| Smoke detection (FIgLib) | **Qwen3-VL-235B-A22B** (`bedrock_qwen.qwen3-vl-235b-a22b`) | Bedrock | `qwen.qwen3-vl-235b-a22b` (420) | 420 | 2026-09-16 |
| Smoke detection (FIgLib) | **claude-opus-4.8** | Gateway | `claude-opus-4.8` (420) | 420 | 2026-09-16 |
| Smoke detection (FIgLib) | **claude-opus-5** | Gateway | `claude-opus-5` (420) | 420 | 2026-09-16 |
| Smoke detection (FIgLib) | **gemini-3.1-pro** | Gateway | `gemini-3.1-pro` (420) | 420 | 2026-09-16 |
| Smoke detection (FIgLib) | **gpt-6-astra** | Gateway | `gpt-6-astra` (420) | 420 | 2026-09-16 |
| Fire danger forecasting (Mesogeos) | **Llama 4 Maverick** (`bedrock_us.meta.llama4-maverick-17b-instruct-v1_0`) | Bedrock | `us.meta.llama4-maverick-17b-instruct-v1:0` (772) | 772 | 2026-09-16 |
| Fire danger forecasting (Mesogeos) | **Qwen3-VL-235B-A22B** (`bedrock_qwen.qwen3-vl-235b-a22b`) | Bedrock | `qwen.qwen3-vl-235b-a22b` (772) | 772 | 2026-09-16 |
| Fire danger forecasting (Mesogeos) | **claude-opus-4.8** | Gateway | `claude-opus-4.8` (772) | 772 | 2026-09-16 |
| Fire danger forecasting (Mesogeos) | **claude-opus-5** | Gateway | `claude-opus-5` (772) | 772 | 2026-09-16 |
| Fire danger forecasting (Mesogeos) | **gemini-3.1-pro** | Gateway | `gemini-3.1-pro` (772) | 772 | 2026-09-16 |
| Fire danger forecasting (Mesogeos) | **gpt-6-astra** | Gateway | `gpt-6-astra` (772) | 772 | 2026-09-16 |

### Summary by Provider and Model

- **Gateway Models**:
  - `claude-opus-4.8`: Served as `claude-opus-4.8` (1,792 total calls across all three tasks).
  - `claude-opus-5`: Served as `claude-opus-5` (1,792 total calls across all three tasks).
  - `gemini-3.1-pro`: Served as `gemini-3.1-pro` (1,792 total calls across all three tasks).
  - `gpt-6-astra`: Served as `gpt-6-astra` (1,792 total calls across all three tasks).
- **Amazon Bedrock Models**:
  - Qwen3-VL-235B-A22B: Served under AWS Bedrock identifier `qwen.qwen3-vl-235b-a22b` (1,792 total calls).
  - Llama 4 Maverick: Served under AWS Bedrock identifier `us.meta.llama4-maverick-17b-instruct-v1:0` (1,792 total calls).

## All Response Files in the Manifest

| Section | Files | Responses | Files with one served identifier |
|---|:---:|:---:|:---:|
| reported | 36 | 10,752 | 36 of 36 |
| tooluse | 78 | 12,168 | 78 of 78 |
| wildfirevqa | 32 | 13,056 | 32 of 32 |
| added_models | 60 | 17,920 | 60 of 60 |
| text_models | 76 | 26,068 | 76 of 76 |

> [!NOTE]
> Every file above carries exactly one served identifier across all of its rows.

## Models Beyond the Six-Model Reported Set

One row per model over the tool-use, aerial, and 2026-09-18 sweep sections; the six reported models appear here with their tool-use and aerial files only.

| Model | Tier | Path | Served identifier (calls) | Tasks in these sections |
|---|:---:|:---:|---|---|
| claude-opus-4.8 | core | Gateway | `claude-opus-4.8` (1,440) | tool use, aerial |
| claude-opus-5 | core | Gateway | `claude-opus-5` (1,440) | tool use, aerial |
| gemini-3.1-pro | core | Gateway | `gemini-3.1-pro` (1,440) | tool use, aerial |
| gpt-6-astra | core | Gateway | `gpt-6-astra` (1,440) | tool use, aerial |
| Nova Lite | added | Bedrock | `amazon.nova-lite-v1:0` (2,920) | allocation, fire danger, smoke, tool use, aerial |
| Nova Pro | added | Bedrock | `amazon.nova-pro-v1:0` (2,920) | allocation, fire danger, smoke, tool use, aerial |
| Nova 2 Lite | added | Bedrock | `us.amazon.nova-2-lite-v1:0` (2,920) | allocation, fire danger, smoke, tool use, aerial |
| Qwen3-VL | core | Bedrock | `qwen.qwen3-vl-235b-a22b` (1,440) | tool use, aerial |
| Llama 4 Maverick | core | Bedrock | `us.meta.llama4-maverick-17b-instruct-v1:0` (1,440) | tool use, aerial |
| Llama 4 Scout | added | Bedrock | `us.meta.llama4-scout-17b-instruct-v1:0` (2,920) | allocation, fire danger, smoke, tool use, aerial |
| Mistral Large 3 | added | Bedrock | `mistral.mistral-large-3-675b-instruct` (2,920) | allocation, fire danger, smoke, tool use, aerial |
| Ministral 3 8B | added | Bedrock | `mistral.ministral-3-8b-instruct` (2,920) | allocation, fire danger, smoke, tool use, aerial |
| Kimi K2.5 | added | Bedrock | `moonshotai.kimi-k2.5` (2,920) | allocation, fire danger, smoke, tool use, aerial |
| Gemma 3 27B | added | Bedrock | `google.gemma-3-27b-it` (2,920) | allocation, fire danger, smoke, tool use, aerial |
| Gemma 3 12B | added | Bedrock | `google.gemma-3-12b-it` (2,920) | allocation, fire danger, smoke, tool use, aerial |
| Gemma 3 4B | added | Bedrock | `google.gemma-3-4b-it` (2,920) | allocation, fire danger, smoke, tool use, aerial |
| Nova Micro | text | Bedrock | `amazon.nova-micro-v1:0` (1,684) | allocation, fire danger, tool use |
| Llama 3.3 70B | text | Bedrock | `us.meta.llama3-3-70b-instruct-v1:0` (1,684) | allocation, fire danger, tool use |
| Llama 3.1 70B | text | Bedrock | `us.meta.llama3-1-70b-instruct-v1:0` (1,684) | allocation, fire danger, tool use |
| Mistral Small 2402 | text | Bedrock | `mistral.mistral-small-2402-v1:0` (1,684) | allocation, fire danger, tool use |
| Devstral 2 123B | text | Bedrock | `mistral.devstral-2-123b` (1,684) | allocation, fire danger, tool use |
| Qwen3 32B | text | Bedrock | `qwen.qwen3-32b-v1:0` (1,684) | allocation, fire danger, tool use |
| Qwen3 Next 80B | text | Bedrock | `qwen.qwen3-next-80b-a3b` (1,684) | allocation, fire danger, tool use |
| Qwen3 Coder 30B | text | Bedrock | `qwen.qwen3-coder-30b-a3b-v1:0` (1,684) | allocation, fire danger, tool use |
| DeepSeek V3.2 | text | Bedrock | `deepseek.v3.2` (1,684) | allocation, fire danger, tool use |
| GPT-OSS 120B | text | Bedrock | `openai.gpt-oss-120b-1:0` (1,684) | allocation, fire danger, tool use |
| GPT-OSS 20B | text | Bedrock | `openai.gpt-oss-20b-1:0` (1,684) | allocation, fire danger, tool use |
| GLM 5 | text | Bedrock | `zai.glm-5` (1,684) | allocation, fire danger, tool use |
| GLM 4.7 | text | Bedrock | `zai.glm-4.7` (1,684) | allocation, fire danger, tool use |
| GLM 4.7 Flash | text | Bedrock | `zai.glm-4.7-flash` (1,684) | allocation, fire danger, tool use |
| MiniMax M2.5 | text | Bedrock | `minimax.minimax-m2.5` (1,684) | allocation, fire danger, tool use |
| Kimi K2 Thinking | text | Bedrock | `moonshot.kimi-k2-thinking` (1,684) | allocation, fire danger, tool use |
| Nemotron Super 3 120B | text | Bedrock | `nvidia.nemotron-super-3-120b` (1,684) | allocation, fire danger, tool use |
| Llama 3.1 8B | text | Bedrock | `us.meta.llama3-1-8b-instruct-v1:0` (1,372) | allocation, fire danger |
| DeepSeek R1 | text | Bedrock | `us.deepseek.r1-v1:0` (1,372) | allocation, fire danger |

