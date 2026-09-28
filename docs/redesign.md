Worth doing. Before I write code, I need to see the *current* state of `swe_agent/` — what the supervisor, roles, and harness actually look like now, so I port forge's lessons into the right places instead of guessing. Let me dig in.
There's a `layers.py` (229 ln) and big `harness.py`/`supervisor.py`. Let me read the core orchestration files to understand the existing 3-layer skeleton.
Good — now I see the current `swe_agent` shape: `layers.py` has L1/L2/L3 but `supervisor.run_agent` is a *flat* `for iteration` loop, and the executor still flattens tool results into `user` messages (the anti-pattern forge fixed). Also `build_system_prompt` still references a single `agent_action` mega-tool. I need the real current tool protocol + harness gate before restructuring. Reading the rest.
Now the critical bit — the actual toolcall protocol in `models.py`/`actions.py` and whether `_run_test_bar` has the forge hardening. Let me read those plus `config.py`.
Now the last pieces — the LLM clients (what `chat_messages` actually returns under native toolcall) and the current `_run_test_bar` (is it hardened?). Reading both.
Let me check the current `_run_test_bar` (is it gameable?) and the config knobs I'll need for the nested loops.
Let me read the actual `_run_test_bar` implementation to see if it's gameable.
I now have the full picture. Key findings:
- `_run_test_bar` (harness.py:640) is **gameable** — only checks pytest exit code, no AST audit (the `assert True` cheat still passes).
- The executor loop **flattens** tool results into `user` messages (the anti-pattern forge fixed) because the LLM clients discard `tool_call_id`.
- `ROLE_TOOLS` already has `tester` (read-only + `finish_verify`) but `finish_verify`/`verify` tools and the `verify.py` module are **only in forge**, not registered in `swe_agent`.
- `analyzer` stage doesn't exist yet (there's `researcher`).

Let me pin down the exact researcher usages and the harness eval functions before I restructure.
I have everything I need. This is a substantial restructure, so let me lay out the plan as tracked tasks, then build it piece by piece — hardening the gate, adding the analyzer stage + planner verify-points, the independent tester, native toolcall protocol, and finally the 3 nested loops.
Starting with the anti-cheat gate (Task 39). I'll add an AST audit to `harness._run_test_bar` so `assert True` / empty tests can't pass.
Now I'll add `ast` import and the audit function, then integrate it into `_run_test_bar`.
Task 39 done. Now Task 36 — rename `researcher` → `analyzer` across registry, models, roles, supervisor.
Now models.py ROLE_ENV / ROLE_FALLBACK_ENV rename:
Now rename in roles.py — function, system prompt, and all internal `finish_research`/`researcher` references:
Fix the leftover `[researcher]` print label:
Now update supervisor.py import/use of `run_researcher` → `run_analyzer`:
Task 36 done. Now Task 37 — planner emits `verify_points`. Extend the planner system prompt, JSON schema, normalization, and persistence.
Now extend the JSON schema and normalization to carry `verify_points`:
Now persist `verify_points` in `_apply_plan` and print them in `run_planner`:
Task 37 done. Now Task 38 — the independent tester. Let me read forge's `verify.py` to port it faithfully into `swe_agent`.
Now Task 40 — native toolcall protocol. First, add `return_meta` to the LM Studio client so `tool_call_id` is preserved:
Now the same `return_meta` for the GLM client:
Now add `chat_toolcalls` to `models.py`:
Task 40's client side is done. Now Task 38 — create `swe_agent/verify.py` (independent tester) using the native protocol:
Now register the `finish_verify` tool (needed by the tester role) in `registry.py`:
Add `verify` to the executor role's toolset:
Now register the `verify` tool in `supervisor.py` (executor's on-demand self-test that invokes the independent tester). I'll add it near the `report` tool:
Task 38 done. Now the big one — Task 41: config knobs + lint + the nested loops. First, add config knobs:
Now add a `run_lint` helper. I'll put it in `harness.py` (it's objective verification, pure code):
I need to import the new lint config in harness.py:
Now the core rewrite. Let me fix the stale `agent_action` instruction in the system prompt, then replace the flat loop with the nested structure. First, the system-prompt text:
I'll continue with Task 41 — rewriting `supervisor.run_agent` into the nested 3-loop structure. Let me first read the current state of the key files.
Let me read the other supporting modules to understand the current architecture before rewriting.
Let me read the remaining key files to understand the full picture.
Now let me read the models, state, and registry modules to understand the toolcall protocol and dispatch.
Let me read the registry module to understand `glm_tools`, `dispatch`, `ROLE_TOOLS`, and the tool execution/context model.
I now have a thorough understanding. Let me check the `LoopCtx` contract and `tools.py` dispatch surface before rewriting.
The architecture is clear. Before rewriting, I spotted a latent bug: `roles.py`'s `run_analyzer` still references `_RESEARCHER_SYSTEM` (renamed to `_ANALYZER_SYSTEM`), which would `NameError` on the first call. Let me verify and check the `return_meta` contract in the LLM clients.
Confirmed: `roles.py` lines 557 and 621 reference the undefined `_RESEARCHER_SYSTEM` (renamed to `_ANALYZER_SYSTEM`) — a `NameError` waiting to fire on the first `run_analyzer` call. Let me fix that and confirm the GLM `return_meta` shape.
GLM's `return_meta` shape matches (`id`/`name`/`arguments`), and `tool_choice=None` correctly maps to `auto` for LM Studio (the qwen-7b executor). Now let me fix the `_RESEARCHER_SYSTEM` `NameError` in `roles.py` and wire up the supervisor.
Now wire up the supervisor: add the `verify` and `ROLE_TOOLS` imports.
Now replace the flat `for iteration` loop in `run_agent` with the nested-loop driver.
Now insert the two nested-loop driver functions before the module-level drift-set comment.
Let me verify the edited `run_agent` tail and the new functions compile correctly.
The structure is intact. Now let me compile all touched modules to catch any syntax errors.
All modules compile. Now let me stub-test the new nested-loop logic end-to-end (Task 42) without needing a live model server — driving `_run_loop3_executor`/`_run_nested_loops` with mocked `chat_toolcalls` and gates.
The managed Python lacks `requests`. Let me set up the isolated venv and install it there, per the runtime isolation rules.
`requests` installed. Now run the stub tests with the venv Python.
All three stub tests pass — the native toolcall protocol, loop_3→lint→pytest→tester flow, loop_1 retry-on-failure, and clean `all_done` termination all work. (Note: only 2 actions dispatched in tests because the real `write_file` hook auto-marks the current task done, which is correct existing behavior.)

Let me confirm what `layers.*` symbols supervisor still depends on, so I don't leave dangling references, and clean up the test artifacts.
`supervisor` only uses `layers.reset_layer_events()` now; the flat `run_l1/2/3` functions in `layers.py` are dormant. Let me confirm nothing else calls them before deciding to retire them, and clean up test artifacts.
`run_l1/2/3_propose/fence/sense` are only defined in `layers.py` and no longer called — the nested loop now lives in `supervisor`. To avoid leaving confusing dead code, I'll update `layers.py`'s docstring to reflect that it now only provides cross-layer telemetry, and mark the flat functions as superseded.
Now re-compile `layers.py` and do a final import sanity check of the whole package.
There's already a sizeable `2026-08-31.md`. Let me check its tail to see what's recorded and whether the nested-loop refactor is captured.
The tail covers a lot of prior work but I need to confirm whether the **forge-absorption nested-loop refactor (Task 41/42)** is actually recorded. Let me search.
The latest forge-absorption nested-loop rewrite (Task 41/42) isn't captured in a dedicated section yet. Let me read the exact code to record it precisely.
The code matches the refactor. Let me check the existing memory section around line 81-100 to avoid duplication and find the right insertion point.
The earlier "三层 loop 重构" section (line 81) documents a *previous* flat L1/L2/L3 design that was then superseded. The latest forge-absorption nested-loop rewrite (Task 41/42) needs its own section. Let me do a fresh compile + import verification in this working session (the prior session's Bash was reportedly broken), then append the memory note.
Verification is clean: `compileall` passes, import OK, both nested-loop functions present, analyzer NameError fix confirmed (`_ANALYZER_SYSTEM` present / `_RESEARCHER_SYSTEM` gone), and flat `layers` functions correctly dormant. Now let me read the exact tail to append a precise memory note.
The file continues past line 162. Let me read to the true end to find the correct append point.
Now I have the true end (line 173). Let me append the Task 41/42 nested-loop refactor note after the architecture-closure section.
Done. The forge-absorption nested-loop refactor (Task 41) and its stub verification (Task 42) were already landed in `swe_agent/supervisor.py` per the prior turn; this session I did a **fresh re-verification** (the prior session's Bash was broken) and **closed the loop by recording it to project memory**.

## What's verified (this session, Bash working)

- `compileall swe_agent` → clean, no errors
- `import swe_agent.supervisor` → OK
- `_run_nested_loops` + `_run_loop3_executor` both present and wired into `run_agent` (replaces the old flat `for iteration` loop)
- Analyzer `NameError` fix confirmed: `_ANALYZER_SYSTEM` exists, the stale `_RESEARCHER_SYSTEM` reference is gone
- `layers.run_l1_propose/run_l2_fence/run_l3_sense` correctly dormant (defined, zero callers — superseded by the nested driver)

## Architecture now in place (your original spec)

```
analyzer → planner(verify_points) → loop_1(attempt)
  └ loop_2(round)
      └ loop_3(step): executor native toolcall (assistant/tool alternation, 1 call/step)
    → lint gate → pytest (single-bar) → tester (independent read-only)
  tester pass ⇒ Agent done; else next attempt; attempts exhausted ⇒ limit_reached
```

Forge lessons absorbed: native OpenAI toolcall protocol (no flattening into `user`), per-action `ToolRegistry.glm_tools("executor")` with `ROLE_TOOLS` allow-list enforcement, `tool_choice="auto"` for qwen-7b, `lint` as 3rd local gate, hardened `test_bar` (AST audit), independent read-only tester consuming `verify_points`.

## Memory written

Appended a `## Task 41/42` section to `~/kuiwork/workdir2/litertlm/.workbuddy/memory/2026-08-31.md` capturing the refactor, the analyzer fix, the stub-test results, and this session's fresh verification — so the canonical architecture is documented for next time.

## Optional loose ends (not done, your call)
1. **Delete the now-dormant `layers.py` flat functions** — currently kept (with a "superseded" docstring) as historical reference. Say the word and I'll remove them for clarity.
2. **Real end-to-end run** with the nested loop already happened in #75 (Conway GOL, local models) — architecture闭环 confirmed; weak executor didn't turn tests all-green (expected, model-capability boundary, not a harness bug).

Want me to purge the dead `layers.py` functions, or run a fresh full e2e through the new `_run_nested_loops` to confirm behavior end-to-end?