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

