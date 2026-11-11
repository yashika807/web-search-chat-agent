"""Persistent, timestamped memory of every question the user has asked.

One JSON object per line (JSONL). Append-only, so a crash can never corrupt
earlier history, and a half-written last line is simply skipped on load.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional


def humanize(delta_seconds: float) -> str:
    """'just now', '5 min ago', '3 hours ago', '2 days ago'..."""
    s = max(0, int(delta_seconds))
    if s < 60:
        return "just now"
    if s < 3600:
        return f"{s // 60} min ago"
    if s < 86400:
        h = s // 3600
        return f"{h} hour{'s' if h != 1 else ''} ago"
    if s < 86400 * 60:
        d = s // 86400
        return f"{d} day{'s' if d != 1 else ''} ago"
    mo = s // (86400 * 30)
    return f"{mo} months ago"


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


class Memory:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # ---- write -----------------------------------------------------------
    def append(
        self,
        question: str,
        answer: str,
        searches: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        ts: Optional[datetime] = None,
    ) -> Dict:
        ts = ts or datetime.now(timezone.utc).astimezone()
        entry = {
            "ts": ts.isoformat(timespec="seconds"),
            "q": question,
            "a": answer,
            "searches": searches or [],
            "sources": sources or [],
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return entry

    def clear(self) -> None:
        if os.path.exists(self.path):
            os.remove(self.path)

    # ---- read ------------------------------------------------------------
    def entries(self) -> List[Dict]:
        out: List[Dict] = []
        if not os.path.exists(self.path):
            return out
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    datetime.fromisoformat(e["ts"])
                    e["q"]
                except (ValueError, KeyError, TypeError):
                    continue  # skip torn / corrupt lines
                out.append(e)
        return out

    def ledger(self, now: datetime, max_entries: int = 120, answer_chars: int = 220) -> str:
        """Compact text block the model reads to recall past conversations.

        Newest last. Each line carries the absolute local timestamp AND a
        relative age, so 'what did I ask yesterday?' needs no date math.
        """
        entries = self.entries()[-max_entries:]
        if not entries:
            return ""
        lines = []
        for e in entries:
            t = datetime.fromisoformat(e["ts"])
            age = humanize((now - t).total_seconds())
            stamp = t.astimezone(now.tzinfo).strftime("%Y-%m-%d %H:%M")
            searched = " [searched web]" if e.get("searches") else ""
            lines.append(
                f"- {stamp} ({age}) Q: {_clip(e['q'], 300)} | A: {_clip(e.get('a', ''), answer_chars)}{searched}"
            )
        return "\n".join(lines)
