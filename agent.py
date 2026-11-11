#!/usr/bin/env python3
"""A fast, grounded chat agent with exactly one tool: web search.

    python agent.py                 # interactive chat
    python agent.py "who won ..."   # one-shot question

See design.md for the reasoning behind every choice here.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

from memory import Memory

DEFAULT_MODEL = "claude-haiku-5-5"  # fast tier; override with --model / AGENT_MODEL
MAX_TOKENS = 1500
MAX_SEARCHES = 3  # per turn: caps latency and cost
MAX_PAUSE_CONTINUATIONS = 3
MAX_SESSION_TURNS = 30  # in-context turns kept live; older ones live in the ledger

WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": MAX_SEARCHES,
}

SYSTEM_PROMPT = """\
You are a fast, accurate chat assistant with one tool: web_search.

SPEED
- Lead with the answer in the first sentence. No preamble, no "let me search", no recap of the question.
- Be concise: a direct answer is usually 1-4 sentences. Add detail only if the question needs it.
- Do NOT search for: greetings, chit-chat, math, code help, stable well-known facts, or questions about this conversation/your memory.
- DO search for: anything recent or that changes (news, prices, scores, versions, who currently holds a role, schedules, availability), niche or specific facts you are not sure of, and anything the user asks you to look up or verify.
- When you search, make ONE precise query first (add the year or "latest" when recency matters). Search again only if the first results do not settle it. Never exceed what is needed.

CORRECTNESS
- Trust search results over your memory for anything time-sensitive. If sources disagree, say so and prefer the most authoritative/primary and most recent one.
- Never invent facts, numbers, quotes, or URLs. If you could not find or verify something, say exactly that, and say what you did find.
- State dates explicitly ("as of 11 Nov 2026") for facts that can go stale.
- Cite the sources you relied on; the search tool attaches citations automatically.

MEMORY
- Every user message starts with a [timestamp] in the user's local time. That is metadata: never repeat it back.
- "CONVERSATION LEDGER" below lists the user's questions from earlier sessions with when they were asked. Together with the messages in this chat, it is your complete memory.
- When asked what they asked / when / "earlier", "yesterday", "last time": answer from the ledger and this chat only, quoting the date/time. Use the relative ages shown; do the date math against the current time given in the first message. If it is not in your memory, say you have no record of it. Do not search the web for it.
- If a new question is a follow-up to something in your memory, use that context.
"""


def build_system(ledger: str) -> List[Dict]:
    """System prompt = static instructions (cached) + per-session ledger (cached)."""
    blocks = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    ledger_text = ledger if ledger else "(empty: this is the user's first session)"
    blocks.append(
        {
            "type": "text",
            "text": f"CONVERSATION LEDGER (earlier sessions, oldest first):\n{ledger_text}",
            "cache_control": {"type": "ephemeral"},
        }
    )
    return blocks


def stamp(now: datetime, text: str) -> str:
    return f"[{now.strftime('%Y-%m-%d %H:%M %Z (%A)')}] {text}"


@dataclass
class Turn:
    answer: str = ""
    searches: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)  # deduped, in citation order


def _collect(content, turn: Turn) -> None:
    for block in content:
        btype = getattr(block, "type", "")
        if btype == "server_tool_use" and getattr(block, "name", "") == "web_search":
            q = (getattr(block, "input", None) or {}).get("query")
            if q:
                turn.searches.append(q)
        elif btype == "text":
            turn.answer += block.text
            for c in getattr(block, "citations", None) or []:
                url = getattr(c, "url", None)
                if url and url not in turn.sources:
                    turn.sources.append(url)


def ask(
    client,
    model: str,
    system: List[Dict],
    history: List[Dict],
    question: str,
    now: datetime,
    on_text: Callable[[str], None] = lambda s: None,
    on_search: Callable[[], None] = lambda: None,
) -> Turn:
    """Run one user turn (streaming). Does not mutate `history`."""
    messages = history[-2 * MAX_SESSION_TURNS :] + [
        {"role": "user", "content": stamp(now, question)}
    ]
    turn = Turn()
    for _ in range(MAX_PAUSE_CONTINUATIONS + 1):
        with client.messages.stream(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=[WEB_SEARCH_TOOL],
            messages=messages,
        ) as stream:
            for event in stream:
                etype = getattr(event, "type", "")
                if etype == "text":
                    on_text(event.text)
                elif etype == "content_block_start" and getattr(
                    event.content_block, "type", ""
                ) == "server_tool_use":
                    on_search()
            final = stream.get_final_message()
        _collect(final.content, turn)
        if final.stop_reason == "pause_turn":
            # Server-side search loop hit its iteration cap; resume where it stopped.
            messages = messages + [{"role": "assistant", "content": final.content}]
            continue
        break
    turn.answer = turn.answer.strip()
    return turn


# --------------------------------------------------------------------------- CLI
def _load_dotenv(path: str) -> None:
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def _now() -> datetime:
    return datetime.now().astimezone()


def _print_history(memory: Memory) -> None:
    entries = memory.entries()
    if not entries:
        print("(no history yet)")
        return
    for e in entries:
        t = datetime.fromisoformat(e["ts"]).strftime("%Y-%m-%d %H:%M")
        print(f"{t}  {e['q']}")


def run_turn(client, model, memory, system, history, question) -> Optional[Turn]:
    import anthropic

    now = _now()
    started = {"text": False}

    def on_text(s: str) -> None:
        started["text"] = True
        sys.stdout.write(s)
        sys.stdout.flush()

    def on_search() -> None:
        sys.stdout.write("\033[2m🔎 searching…\033[0m\n" if sys.stdout.isatty() else "[searching]\n")
        sys.stdout.flush()

    try:
        turn = ask(client, model, system, history, question, now, on_text, on_search)
    except KeyboardInterrupt:
        print("\n(interrupted)")
        return None
    except anthropic.APIError as e:
        print(f"\n[error] {getattr(e, 'message', e)}", file=sys.stderr)
        return None

    if not turn.answer:
        turn.answer = "(no answer returned)"
        print(turn.answer)
    if turn.sources:
        print("\n\nSources:")
        for i, u in enumerate(turn.sources[:5], 1):
            print(f"  [{i}] {u}")
    print()
    history.append({"role": "user", "content": stamp(now, question)})
    history.append({"role": "assistant", "content": turn.answer})
    memory.append(question, turn.answer, turn.searches, turn.sources, now)
    return turn


def main(argv: Optional[List[str]] = None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    _load_dotenv(os.path.join(here, ".env"))

    p = argparse.ArgumentParser(description="Chat agent with web search + timestamped memory")
    p.add_argument("question", nargs="*", help="ask once and exit (omit for interactive chat)")
    p.add_argument("--model", default=os.environ.get("AGENT_MODEL", DEFAULT_MODEL))
    p.add_argument("--memory", default=os.path.join(here, "memory", "history.jsonl"))
    args = p.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY (env var or a .env file next to agent.py).", file=sys.stderr)
        return 2

    import anthropic

    client = anthropic.Anthropic()
    memory = Memory(args.memory)
    system = build_system(memory.ledger(_now()))
    history: List[Dict] = []

    if args.question:
        return 0 if run_turn(client, args.model, memory, system, history, " ".join(args.question)) else 1

    try:
        import readline  # noqa: F401  (line editing / arrow keys)
    except ImportError:
        pass

    n = len(memory.entries())
    print(f"Chat agent · {args.model} · {n} past question{'s' if n != 1 else ''} remembered")
    print("Commands: /history  /forget  /exit\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            continue
        if q in ("/exit", "/quit"):
            return 0
        if q == "/history":
            _print_history(memory)
            continue
        if q == "/forget":
            if input("Erase ALL saved history? (yes/no) ").strip().lower() == "yes":
                memory.clear()
                history.clear()
                system = build_system("")
                print("History erased.")
            continue
        print("agent> ", end="", flush=True)
        run_turn(client, args.model, memory, system, history, q)


if __name__ == "__main__":
    raise SystemExit(main())
