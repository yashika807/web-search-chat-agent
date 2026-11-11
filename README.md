# Web-search chat agent

A command-line chat agent with a single tool, web search. It answers fast, cites sources, and
remembers every question you asked and when, across sessions. See **[design.md](design.md)** for the
design and the reasoning behind it.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...      # or copy .env.example to .env
python agent.py                          # interactive
python agent.py "latest stable Python version?"   # one-shot
```

In chat: `/history` lists everything you've asked and when, `/forget` erases it, `/exit` quits.
Try: ask something current, quit, restart, then ask *"what did I ask you earlier?"*

## Test (offline, no API key)

```bash
python -m unittest discover -s tests
```

Model defaults to `claude-haiku-5-5`; change with `--model` or `AGENT_MODEL`.
