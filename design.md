# Design

A chat agent with **one tool (web search)** that is built around three goals from the brief:
**answer fast, get things right, remember what you asked and when.**

## Architecture

```
 you ──► agent.py ──► Claude (streaming) ──► web_search (server-side, ≤3/turn)
            │  ▲
            │  └── system prompt = rules + CONVERSATION LEDGER (from memory.py)
            ▼
      memory/history.jsonl   (append-only: timestamp, question, answer, queries, sources)
```

Two small files: `agent.py` (loop, prompt, CLI) and `memory.py` (storage + ledger). No framework,
no database, no vector store. The whole thing is about 300 lines so every behaviour is auditable.

## 1. Fast

| Decision | Why |
|---|---|
| Streaming output | First tokens appear immediately; perceived latency is time-to-first-token, not time-to-last. |
| Fast-tier model by default (`claude-haiku-5-5`, switch with `--model`) | Most turns are "search, read, summarise": a small model is enough and several times faster. |
| Search is **server-side** and capped at 3 per turn | One round trip to the API instead of a client-side tool loop that calls the model, then a search API, then the model again. The cap bounds worst-case latency and cost. |
| Prompt says *when not to search* | Greetings, math, code, stable facts and memory questions are answered with zero search latency. This is the single biggest speed lever. |
| "One precise query first" | Most questions are settled by one search; extra searches only if the first did not settle it. |
| "Answer first, no preamble" + concise default | Fewer tokens generated = faster and easier to read. |
| Prompt caching on the system prompt and ledger | Static rules and the (growing) ledger are cached, so each turn only pays for the new message. |
| Session history trimmed to the last 30 turns | Keeps per-turn input small; older turns are still available through the ledger. |

## 2. Correct

| Decision | Why |
|---|---|
| Current date/time in every user message | The model cannot know "today"; without it, "latest" and "this year" queries silently use stale training data. |
| "Trust search results over memory for anything time-sensitive" | Resolves the classic conflict between stale weights and fresh evidence in favour of the evidence. |
| Sources are cited and printed (deduped, in citation order) | The user can verify any claim in one click; citations come from the search tool, not from the model's imagination. |
| "Never invent facts, numbers, quotes, URLs. Say what you could not verify." | Prefers an honest "I couldn't confirm X, here is what I found" to a confident wrong answer. |
| Dates on perishable facts ("as of 11 Nov 2026") | Makes staleness visible. |
| Conflicting sources are surfaced, authoritative + recent wins | The model reports disagreement instead of silently picking one. |

## 3. Remembers what you asked, and when

Memory is deliberately simple and exact, because "when" must never be hallucinated.

* **Every turn is written to `memory/history.jsonl`** (append-only JSONL, fsync'd): ISO timestamp with
  timezone, the question, the answer, the search queries run, and the source URLs.
  Append-only means a crash cannot corrupt history; a torn last line is skipped on load.
* **Every user message is prefixed with `[YYYY-MM-DD HH:MM TZ (Weekday)]`**, so inside a session the
  model sees real timestamps for every question and the current time is always known.
* **At startup the ledger is injected into the system prompt**: one line per past question with the
  absolute time *and* a precomputed relative age:
  `- 2026-11-10 12:00 (1 day ago) Q: what is rust? | A: A systems language… [searched web]`.
  Precomputed ages mean "what did I ask yesterday?" needs no date arithmetic from the model.
* Recall questions are answered **from memory only, never the web**; if it is not in memory the agent
  says it has no record rather than guessing.
* Ledger keeps the newest 120 entries with answers clipped to ~220 chars, bounding prompt size.
* Why not a vector DB / summaries? For a single user's chat history the full ledger fits in context,
  is cached, and is lossless for the two things asked for: *what* and *when*. Retrieval would add
  latency and a way to miss the exact question. If history grew into the thousands, the next step
  would be the same ledger plus embedding search over older entries.
* `/history` prints the raw log deterministically (no model involved); `/forget` erases it (with
  confirmation). Memory is local and git-ignored.

## Failure handling

* API errors are shown and the turn is **not** saved, so memory never contains a half-answer.
* `pause_turn` (server-side search loop hit its iteration limit) is resumed automatically.
* Ctrl-C during a response cancels the turn cleanly.

## Testing

`tests/test_agent.py` (offline, no API key needed) uses a fake streaming client to check: exactly one
tool is exposed; answer/queries/sources are collected and deduped; text is streamed; `pause_turn`
resumes; timestamps are prefixed; system blocks are cached and contain the ledger; history is trimmed
without mutation; and the memory file round-trips, caps, and survives corrupt lines.

Run: `python -m unittest discover -s tests`

## Trade-offs / what I'd do next

* Haiku-tier is fastest; for hard multi-hop research `--model` can be raised at the cost of latency.
* Ledger answers are clipped, so recall of *what the answer was* is approximate; recall of the
  question and its time is exact. The full answer is in the JSONL if needed.
* Natural next steps: an eval set of fresh-fact questions scored for correctness and latency, and
  embedding retrieval over old history once it no longer fits in the ledger.
