#!/usr/bin/env python3
"""
Run a scripted conversation scenario against Claude or a local SLM.

A scenario is a YAML file listing the turns to send in order.
The conversation history is carried forward across turns (multi-turn).

Usage
-----
    python3 run_scenario.py scenarios/sycophancy_basic.yaml
    python3 run_scenario.py scenarios/authority_pressure.yaml --provider slm
    python3 run_scenario.py scenarios/my_demo.yaml --model mistral --provider slm

Scenario file format (YAML)
----------------------------
    name: "My demo"
    system: "Optional system prompt"   # omit for no system prompt
    turns:
      - "First user message"
      - "Second user message"
      - "Third user message"
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

try:
    import yaml
except ImportError:
    sys.exit("PyYAML not found — run: pip3 install PyYAML")


# ── Provider backends ─────────────────────────────────────────────────────────

def ask_claude(client, model: str, system: str | None, messages: list[dict], max_tokens: int) -> str:
    kwargs = dict(model=model, max_tokens=max_tokens, messages=messages)
    if system:
        kwargs["system"] = system
    r = client.messages.create(**kwargs)
    return r.content[0].text.strip()


def ask_ollama(model: str, system: str | None, messages: list[dict], max_tokens: int) -> str:
    import ollama
    payload = []
    if system:
        payload.append({"role": "system", "content": system})
    payload.extend(messages)
    resp = ollama.chat(
        model=model,
        messages=payload,
        options={"num_predict": max_tokens, "temperature": 0.7},
    )
    return resp["message"]["content"].strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("scenario",             help="Path to scenario YAML file")
    parser.add_argument("--provider", default="claude", choices=["claude", "slm"])
    parser.add_argument("--model",    default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    scenario_path = Path(args.scenario)
    if not scenario_path.exists():
        sys.exit(f"Scenario file not found: {scenario_path}")

    with open(scenario_path) as f:
        scenario = yaml.safe_load(f)

    name   = scenario.get("name", scenario_path.stem)
    system = scenario.get("system") or None
    turns  = scenario.get("turns", [])

    if not turns:
        sys.exit("Scenario has no turns.")

    # Set up provider
    if args.provider == "claude":
        try:
            import anthropic
        except ImportError:
            sys.exit("anthropic package not found — run: pip3 install anthropic")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            sys.exit("ANTHROPIC_API_KEY not set — add it to .env")
        model = args.model or "claude-haiku-4-5-20251001"
        client = anthropic.Anthropic(api_key=api_key)
        ask = lambda msgs, mt: ask_claude(client, model, system, msgs, mt)
    else:
        try:
            import ollama
        except ImportError:
            sys.exit("ollama package not found — run: pip3 install ollama")
        model = args.model or "llama3.2"
        try:
            resp = ollama.list()
            raw = resp.models if hasattr(resp, "models") else resp.get("models", [])
            available = [
                (getattr(m, "model", None) or m.get("name", "")).split(":")[0]
                for m in raw
            ]
        except Exception as e:
            sys.exit(f"Cannot reach Ollama — is it running?  ollama serve\n{e}")
        if model not in available:
            sys.exit(f"Model '{model}' not pulled.\nRun: ollama pull {model}")
        ask = lambda msgs, mt: ask_ollama(model, system, msgs, mt)

    # ── Run scenario ──────────────────────────────────────────────────────────
    print(f"Scenario : {name}")
    print(f"Provider : {args.provider.upper()} / {model}")
    if system:
        print(f"System   : {system[:80]}")
    print("=" * 72)

    history: list[dict] = []

    for i, turn_text in enumerate(turns, 1):
        print(f"\n[Turn {i}] You: {turn_text}")
        print("-" * 60)

        history.append({"role": "user", "content": turn_text})
        reply = ask(history, args.max_tokens)
        history.append({"role": "assistant", "content": reply})

        print(f"{model}: {reply}")

    print("\n" + "=" * 72)
    print(f"Scenario complete — {len(turns)} turns")


if __name__ == "__main__":
    main()
