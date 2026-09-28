#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the unified hook bus (hooks.py) and the named-limit guard (guard.py).

Refactor #27 (2026-09-12): replaced magic-string phases/scopes with first-class
HookPoint / GateAction / GateDecision and a single HookHub, and moved loop fences
into a self-describing Guard whose reset timing is declared via `reset_at: HookPoint`.

These tests are model-free / tool-free and exercise the wiring directly so that a
regression in dispatch priority or reset routing turns the suite red.
"""

import pytest

from swe_agent.hooks import (
    HOOK_HUB,
    GateAction,
    GateDecision,
    HookHub,
    HookPayload,
    HookPoint,
    _legacy_adapter,
)
from swe_agent.guard import Guard, Limit


# ---------------------------------------------------------------------------
# HookPoint
# ---------------------------------------------------------------------------

def test_hookpoint_values_are_unique():
    vals = [p.value for p in HookPoint]
    assert len(vals) == len(set(vals)), "HookPoint must have distinct string values"


def test_hookpoint_loop_start_anchors_exist():
    for p in (HookPoint.L1_LOOP_START, HookPoint.L2_LOOP_START, HookPoint.L3_LOOP_START):
        assert p in HookPoint
    assert HookPoint.RUN_START != HookPoint.L1_LOOP_START


# ---------------------------------------------------------------------------
# HookHub: subscribe + emit
# ---------------------------------------------------------------------------

def test_hookhub_on_emit_fires_subscriber():
    hub = HookHub()
    seen = []

    def fn(payload: HookPayload):
        seen.append(payload.fields)
        return None

    hub.on(HookPoint.BEFORE_TOOL_CALL, fn)
    assert fn not in hub.subscribers(HookPoint.AFTER_TOOL_CALL)
    hub.emit(HookPoint.BEFORE_TOOL_CALL, ctx="C", tool="read_file")
    assert len(seen) == 1
    assert seen[0]["ctx"] == "C"
    assert seen[0]["tool"] == "read_file"


def test_hookhub_emit_runs_all_subscribers_before_verdict():
    """Invariant: a blocking hook must NOT prevent earlier subscribers' side effects.

    Mirrors test_registry_dispatch's "global before still runs before intercept".
    """
    hub = HookHub()
    counter = {"n": 0}

    def counter_fn(payload: HookPayload):
        counter["n"] += 1
        return None  # side-effect only (ALLOW)

    def blocker(payload: HookPayload):
        return "blocked: not allowed"

    hub.on(HookPoint.BEFORE_TOOL_CALL, counter_fn)
    hub.on(HookPoint.BEFORE_TOOL_CALL, blocker)

    decisions = hub.emit(HookPoint.BEFORE_TOOL_CALL, tool="x")
    # counter subscriber ran even though a later subscriber blocked
    assert counter["n"] == 1
    assert HookHub.first_block(decisions) == "blocked: not allowed"


def test_hookhub_fail_open_on_subscriber_exception():
    hub = HookHub()
    hits = []

    def boom(payload: HookPayload):
        raise RuntimeError("kaboom")

    def ok(payload: HookPayload):
        hits.append(payload.point)
        return None

    hub.on(HookPoint.AFTER_TOOL_CALL, boom)
    hub.on(HookPoint.AFTER_TOOL_CALL, ok)

    out = hub.emit(HookPoint.AFTER_TOOL_CALL, result="r")
    # ok still ran; boom failed open and contributed None, no exception escapes
    assert hits == [HookPoint.AFTER_TOOL_CALL]
    assert out == [None, None]


def test_hookhub_clear_removes_subscribers():
    hub = HookHub()

    def fn(payload: HookPayload):
        return None

    hub.on(HookPoint.BEFORE_TOOL_CALL, fn)
    assert len(hub.subscribers(HookPoint.BEFORE_TOOL_CALL)) == 1
    hub.clear(HookPoint.BEFORE_TOOL_CALL)
    assert len(hub.subscribers(HookPoint.BEFORE_TOOL_CALL)) == 0

    hub.on(HookPoint.AFTER_TOOL_CALL, fn)
    hub.clear()
    assert len(hub.subscribers(HookPoint.AFTER_TOOL_CALL)) == 0


# ---------------------------------------------------------------------------
# GateDecision + verdict helpers
# ---------------------------------------------------------------------------

def test_gate_decision_factories():
    assert GateDecision.allow().is_allow
    assert GateDecision.allow().action == GateAction.ALLOW

    r = GateDecision.reject("bad input")
    assert r.action == GateAction.REJECT and r.reason == "bad input" and not r.is_allow

    b = GateDecision.break_loop("stuck")
    assert b.action == GateAction.BREAK_LOOP and b.reason == "stuck"

    o = GateDecision.override("replaced")
    assert o.action == GateAction.OVERRIDE and o.reason == "replaced"


def test_first_block_priority_first_non_allow_wins():
    # legacy str blocks
    assert HookHub.first_block([None, "stop", None]) == "stop"
    # GateDecision REJECT/BREAK_LOOP blocks; ALLOW passes
    d = [
        GateDecision.allow(),
        GateDecision.reject("nope"),
        GateDecision.break_loop("loop"),
    ]
    assert HookHub.first_block(d) == "nope"
    # all allow -> None
    assert HookHub.first_block([None, GateDecision.allow(), None]) is None
    # empty -> None
    assert HookHub.first_block([]) is None
    # block with empty reason falls back to a default label
    assert HookHub.first_block([GateDecision.reject("")]) == "rejected_by_hook"


def test_override_of_takes_last_override():
    original = "ORIGINAL"
    # last str wins
    assert HookHub.override_of([None, "A", "B"], original) == "B"
    # GateDecision.OVERRIDE explicit
    assert HookHub.override_of([None, GateDecision.override("X")], original) == "X"
    # no override -> original preserved
    assert HookHub.override_of([None, GateDecision.allow()], original) == original
    assert HookHub.override_of([], original) == original


# ---------------------------------------------------------------------------
# _legacy_adapter: old fn(ctx, params, result) -> Optional[str]
# ---------------------------------------------------------------------------

def test_legacy_adapter_routes_fields():
    captured = {}

    def old_fn(ctx, params, result):
        captured["ctx"] = ctx
        captured["params"] = params
        captured["result"] = result
        return "legacy-block"

    sub = _legacy_adapter(old_fn)
    ret = sub(HookPayload(point=HookPoint.BEFORE_TOOL_CALL,
                          fields={"ctx": "C", "params": {"a": 1}, "result": "R"}))
    assert ret == "legacy-block"
    assert captured == {"ctx": "C", "params": {"a": 1}, "result": "R"}


# ---------------------------------------------------------------------------
# Guard: reset_at routing + counting
# ---------------------------------------------------------------------------

def _sample_guard():
    return Guard([
        Limit("run_level", reset_at=HookPoint.RUN_START, threshold=2),
        Limit("round_level", reset_at=HookPoint.L2_LOOP_START, threshold=3),
        Limit("step_level", reset_at=HookPoint.L3_LOOP_START, threshold=0),
    ])


def test_guard_tick_and_tripped():
    g = _sample_guard()
    assert g.value("run_level") == 0
    g.tick("run_level")
    assert g.value("run_level") == 1
    assert not g.tripped("run_level")
    g.tick("run_level")
    assert g.tripped("run_level")
    # threshold 0 never trips
    g.tick("step_level")
    assert not g.tripped("step_level")


def test_guard_tick_undeclared_raises():
    g = _sample_guard()
    with pytest.raises(KeyError):
        g.tick("not_declared")


def test_reset_at_only_resets_matching_anchor():
    g = _sample_guard()
    g.tick("run_level")
    g.tick("round_level")
    g.tick("step_level")
    # resetting L2 must NOT touch RUN_START / L3 limits
    hit = g.reset_at(HookPoint.L2_LOOP_START)
    assert hit == ("round_level",)
    assert g.value("round_level") == 0
    assert g.value("run_level") == 1, "RUN_START limit must survive an L2 reset"
    assert g.value("step_level") == 1, "L3 limit must survive an L2 reset"


def test_reset_at_run_start_clears_run_anchored_limits():
    g = _sample_guard()
    g.tick("run_level")
    g.tick("round_level")
    hit = g.reset_at(HookPoint.RUN_START)
    assert "run_level" in hit
    assert g.value("run_level") == 0
    assert g.value("round_level") == 1, "L2-anchored limit must survive a RUN_START reset"


def test_reset_all_clears_everything():
    g = _sample_guard()
    g.tick("run_level")
    g.tick("round_level")
    g.tick("step_level")
    hit = g.reset_all()
    assert set(hit) == {"run_level", "round_level", "step_level"}
    assert all(g.value(n) == 0 for n in hit)


def test_limit_reset_at_must_be_hookpoint():
    with pytest.raises(TypeError):
        Limit("bad", reset_at="run_start")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Module singleton smoke: HOOK_HUB is a HookHub and starts empty-ish
# ---------------------------------------------------------------------------

def test_global_hook_hub_is_singleton_hub():
    assert isinstance(HOOK_HUB, HookHub)
    # exercise it without leaking state across tests
    HOOK_HUB.clear()
    for p in HookPoint:
        assert HOOK_HUB.subscribers(p) == ()
