#!/usr/bin/env python3
"""
Ad hoc interactive chat with a local SLM via Ollama.

Quick start
-----------
    brew install ollama && ollama serve          # install and start Ollama
    ollama pull llama3.2                         # pull a model (~2 GB)
    python run_adhoc_slm.py                      # start chatting

Usage
-----
    python run_adhoc_slm.py                                   # llama3.2, interactive REPL
    python run_adhoc_slm.py --model mistral                   # use Mistral
    python run_adhoc_slm.py --system "You are..."             # custom system prompt
    python run_adhoc_slm.py --once "What is 2+2?"             # one-shot, exit immediately
    python run_adhoc_slm.py --no-stream                       # collect full reply before printing
    python run_adhoc_slm.py --thinking                        # show <think> scratchpad
    python run_adhoc_slm.py --model deepseek-r1 --thinking    # deepseek-r1 has rich scratchpad

REPL commands
-------------
    /exit   or  /quit      quit
    /clear                 reset conversation history
    /system <text>         change system prompt mid-session
    /models                list locally available models
"""

import argparse
import re
import sys
import textwrap

try:
    import ollama
except ImportError:
    sys.exit("ollama package not found — run: pip install ollama")

_THINK_WIDTH = 68


def _available_models() -> list[str]:
    resp = ollama.list()
    models = resp.models if hasattr(resp, "models") else resp.get("models", [])
    return [getattr(m, "model", None) or m.get("name", "") for m in models]


def _check_model(model: str) -> None:
    try:
        models = _available_models()
    except Exception as e:
        sys.exit(f"Cannot reach Ollama — is it running?  Start it with: ollama serve\n{e}")

    base_names = [m.split(":")[0] for m in models]
    if model not in base_names and model not in models:
        print(f"Model '{model}' not found locally.")
        print(f"Available: {', '.join(models) or '(none — pull one first)'}")
        print(f"Pull it with: ollama pull {model}")
        sys.exit(1)


def _print_thinking(think_text: str) -> None:
    print(f"  ┌─ Thinking {'─' * (_THINK_WIDTH - 10)}┐")
    for raw_line in think_text.splitlines():
        for line in textwrap.wrap(raw_line, _THINK_WIDTH - 4) or [""]:
            print(f"  │ {line:<{_THINK_WIDTH - 4}} │")
    print(f"  └{'─' * _THINK_WIDTH}┘")
    print()


def _split_thinking(content: str) -> tuple[str | None, str]:
    """Extract <think>…</think> block; return (think_text, answer)."""
    m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if m:
        think_text = m.group(1).strip()
        answer = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return think_text, answer
    return None, content


def chat(model: str, system: str | None, messages: list[dict], stream: bool, thinking: bool) -> str:
    payload = []
    if system:
        payload.append({"role": "system", "content": system})
    payload.extend(messages)

    # When showing thinking, buffer the full response so we can split out <think> tags.
    # Streaming + thinking: stream silently, then display think block + answer.
    if thinking:
        if stream:
            chunks: list[str] = []
            for part in ollama.chat(model=model, messages=payload, stream=True):
                chunks.append(part["message"]["content"])
            full = "".join(chunks)
        else:
            resp = ollama.chat(model=model, messages=payload)
            full = resp["message"]["content"]

        think_text, answer = _split_thinking(full)
        if think_text:
            _print_thinking(think_text)
        else:
            print("  (no <think> block — model may not support scratchpad output)")
            print()
        print(answer)
        return answer

    if stream:
        chunks = []
        for part in ollama.chat(model=model, messages=payload, stream=True):
            chunk = part["message"]["content"]
            print(chunk, end="", flush=True)
            chunks.append(chunk)
        print()
        return "".join(chunks)
    else:
        resp = ollama.chat(model=model, messages=payload)
        reply: str = resp["message"]["content"]
        print(reply)
        return reply


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model",     default="llama3.2", help="Ollama model name (default: llama3.2)")
    parser.add_argument("--system",    default=None,        help="System prompt")
    parser.add_argument("--once",      default=None,        help="Send one message and exit")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument(
        "--thinking", action="store_true",
        help="Show model scratchpad — parses <think>…</think> tags emitted by models like deepseek-r1",
    )
    args = parser.parse_args()

    _check_model(args.model)

    if args.thinking and args.model == "llama3.2":
        print("Note: llama3.2 doesn't emit <think> tags. For a visible scratchpad, try:")
        print("  ollama pull deepseek-r1   then   --model deepseek-r1 --thinking")
        print()

    stream = not args.no_stream
    system: str | None = args.system

    print(f"Model    : {args.model}  [local / Ollama]")
    print(f"Thinking : {'ON — scratchpad shown before each reply' if args.thinking else 'off'}")
    if system:
        print(f"System   : {system[:80]}")
    print("=" * 60)

    history: list[dict] = []

    # ── One-shot mode ─────────────────────────────────────────────────────────
    if args.once:
        history.append({"role": "user", "content": args.once})
        print(f"\n{args.model}:")
        chat(args.model, system, history, stream, args.thinking)
        return

    # ── Interactive REPL ──────────────────────────────────────────────────────
    print("Commands: /exit · /clear · /system <text> · /models")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[exit]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/exit", "/quit"):
            break

        if user_input.lower() == "/clear":
            history.clear()
            print("[conversation cleared]")
            continue

        if user_input.lower() == "/models":
            try:
                print("  " + "\n  ".join(_available_models()))
            except Exception:
                print("[cannot reach Ollama]")
            continue

        if user_input.lower().startswith("/system "):
            system = user_input[8:].strip()
            print(f"[system prompt updated: {system[:60]}]")
            continue

        history.append({"role": "user", "content": user_input})
        print(f"\n{args.model}:")
        reply = chat(args.model, system, history, stream, args.thinking)
        history.append({"role": "assistant", "content": reply})
        print()


if __name__ == "__main__":
    main()
