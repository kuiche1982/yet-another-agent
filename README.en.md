# litertlm

[中文](./README.md) | **English**

A personal learning project: hands-on implementation of agent-harness techniques using local models (litert-lm / MLX quantized). It explores SWE-agent-style task orchestration, context management, memory, evaluation, and plugin architecture. **Experimental — for reference only. Not guaranteed to work; not for production or critical tasks.**

## What it is

A software-engineering agent harness built from scratch. Core ideas:

- **Orchestration**: `swe_agent/supervisor.py` assembles three nested loops (L1 attempt → L2 round → L3 step), with gates at each level (pytest / tester / lint) to drive progress and rollback.
- **Hooks & guards**: `swe_agent/hooks.py` provides a unified event bus (HOOK_HUB); guards can be attached before/after tool calls (write/read size, stall, unsolvable, etc.) to intercept and rewrite tool behavior.
- **Tool registry**: `swe_agent/tools.py` ToolRegistry handles tool dispatch and safety boundaries.
- **Context management** (`contextmgr/`): compression, distillation, chunking, retrieval, and RAG injection to bound context cost in long tasks.
- **Memory** (`memory/`): incremental recursive summary tree, memory classification, BGE-M3 vector retrieval, persistence.
- **Evaluation** (`judge/` + `tests/verify_*.py`): probes and regressions for intent detection, PII detection, structured output, LSP integration, etc.
- **Plugins** (`plugins/`): Claude Code-style config-only / tool plugins (lsp, ocr, web_fetch, websearch).
- **Voice input** (`voiceinput/`): macOS recording → whisper.cpp transcription for unattended pipelines.

## Layout

```
swe_agent/    harness core (supervisor, layers, guards, tools, roles, plugins, skills, LSP, MCP)
contextmgr/   context management & RAG injection
memory/       memory system (summary tree, classification, retrieval)
judge/        intent / boundary / probe evaluation scripts
plugins/      plugins (lsp, ocr, web_fetch, websearch)
scripts/      experiments & end-to-end evals (fib, conway, go bench, e2e, ...)
docs/         architecture & experiment notes (code is source of truth)
tests/        pytest cases + verify_*.py probes
examples/     Go examples (game of life, todo)
voiceinput/   macOS voice-input transcription tool
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest            # run tests (tests/ + verify_*.py)
```

Requires local LLM inference (litert-lm 0.15 / MLX quantized models). API keys and model paths are injected via `.env` (gitignored); only placeholders are kept in code.

## Notes

- This is **learning-notes-grade code**: many decisions exist to validate "whether approach X works", not to be production-ready.
- Some scripts and docs carry obvious experimental traces (one-off scripts, ad-hoc eval records). Use selectively.
- For design rationale, see `docs/ARCHITECTURE.md` and `docs/harness_design_map.md`.

## License

See [LICENSE](./LICENSE).
