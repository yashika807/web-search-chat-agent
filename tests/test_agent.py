import os, sys, tempfile, unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agent
from memory import Memory, humanize

TZ = timezone(timedelta(hours=5, minutes=30))
NOW = datetime(2026, 11, 11, 14, 0, tzinfo=TZ)


class FakeStream:
    """Mimics anthropic's messages.stream() context manager."""
    def __init__(self, events, final):
        self.events, self.final = events, final
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __iter__(self): return iter(self.events)
    def get_final_message(self): return self.final


class FakeClient:
    def __init__(self, streams):
        self.streams, self.calls = list(streams), []
        self.messages = NS(stream=self._stream)
    def _stream(self, **kw):
        self.calls.append(kw)
        return self.streams.pop(0)


def search_stream():
    cite = NS(url="https://example.com/a")
    final = NS(stop_reason="end_turn", content=[
        NS(type="server_tool_use", name="web_search", input={"query": "x 2026"}),
        NS(type="web_search_tool_result"),
        NS(type="text", text="It is 42.", citations=[cite, cite]),
    ])
    events = [NS(type="content_block_start", content_block=NS(type="server_tool_use")),
              NS(type="text", text="It is 42.")]
    return FakeStream(events, final)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "h.jsonl")
        self.m = Memory(self.path)

    def test_roundtrip_and_ledger_has_time_and_age(self):
        self.m.append("what is rust?", "A language.", ts=NOW - timedelta(days=1, hours=2))
        self.m.append("btc price?", "$1", searches=["btc"], ts=NOW - timedelta(minutes=5))
        led = self.m.ledger(NOW)
        self.assertIn("2026-11-10 12:00 (1 day ago) Q: what is rust?", led)
        self.assertIn("(5 min ago) Q: btc price?", led)
        self.assertIn("[searched web]", led.splitlines()[1])
        self.assertNotIn("[searched web]", led.splitlines()[0])

    def test_corrupt_and_torn_lines_skipped(self):
        self.m.append("ok", "fine", ts=NOW)
        with open(self.path, "a") as f:
            f.write("not json\n{\"ts\": \"bad\", \"q\": 1}\n{\"ts\": \"2026-11-11T1")
        self.assertEqual([e["q"] for e in self.m.entries()], ["ok"])

    def test_empty_ledger_and_clear(self):
        self.assertEqual(self.m.ledger(NOW), "")
        self.m.append("q", "a", ts=NOW); self.m.clear()
        self.assertEqual(self.m.entries(), [])

    def test_ledger_cap_keeps_newest(self):
        for i in range(10):
            self.m.append(f"q{i}", "a", ts=NOW)
        led = self.m.ledger(NOW, max_entries=3)
        self.assertEqual(len(led.splitlines()), 3)
        self.assertIn("q9", led); self.assertNotIn("q6", led)

    def test_humanize(self):
        self.assertEqual(humanize(5), "just now")
        self.assertEqual(humanize(3600), "1 hour ago")
        self.assertEqual(humanize(86400 * 3), "3 days ago")


class AgentTests(unittest.TestCase):
    def test_ask_collects_answer_searches_sources_and_streams(self):
        c = FakeClient([search_stream()])
        out, searched = [], []
        t = agent.ask(c, "m", agent.build_system(""), [], "meaning?", NOW,
                      on_text=out.append, on_search=lambda: searched.append(1))
        self.assertEqual(t.answer, "It is 42.")
        self.assertEqual(t.searches, ["x 2026"])
        self.assertEqual(t.sources, ["https://example.com/a"])  # deduped
        self.assertEqual(out, ["It is 42."]); self.assertEqual(searched, [1])
        kw = c.calls[0]
        self.assertEqual(kw["tools"], [agent.WEB_SEARCH_TOOL])  # exactly one tool
        self.assertEqual(len(kw["tools"]), 1)
        self.assertTrue(kw["messages"][-1]["content"].startswith("[2026-11-11 14:00 UTC+05:30"))

    def test_pause_turn_resumes(self):
        paused = FakeStream([], NS(stop_reason="pause_turn", content=[NS(type="text", text="Hel", citations=None)]))
        done = FakeStream([], NS(stop_reason="end_turn", content=[NS(type="text", text="lo", citations=None)]))
        c = FakeClient([paused, done])
        t = agent.ask(c, "m", agent.build_system(""), [], "hi", NOW)
        self.assertEqual(t.answer, "Hello")
        self.assertEqual(c.calls[1]["messages"][-1]["role"], "assistant")

    def test_system_contains_ledger_and_is_cached(self):
        s = agent.build_system("- 2026-11-10 09:00 (1 day ago) Q: hi | A: yo")
        self.assertIn("Q: hi", s[1]["text"])
        self.assertTrue(all("cache_control" in b for b in s))

    def test_history_is_trimmed_but_not_mutated(self):
        hist = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(200)]
        c = FakeClient([search_stream()])
        agent.ask(c, "m", agent.build_system(""), hist, "q", NOW)
        self.assertEqual(len(hist), 200)
        self.assertEqual(len(c.calls[0]["messages"]), 2 * agent.MAX_SESSION_TURNS + 1)


if __name__ == "__main__":
    unittest.main()
