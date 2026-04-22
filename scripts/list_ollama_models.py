#!/usr/bin/env python3
"""Lista modelos instalados no Ollama: tenta o container Docker, senao o host
(127.0.0.1:11434)."""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.request

from common.config_loader import resolve_ollama_model

OLLAMA_CONTAINER = "tc-fiap-ollama"
HOST_TAGS_URL = "http://127.0.0.1:11434/api/tags"


def _from_docker() -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["docker", "exec", OLLAMA_CONTAINER, "ollama", "list"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.returncode == 0:
            return True, proc.stdout.strip() or "(lista vazia)"
        return False, proc.stderr.strip() or f"exit {proc.returncode}"
    except FileNotFoundError:
        return False, "comando docker nao encontrado"
    except subprocess.TimeoutExpired:
        return False, "timeout"


def _from_host_tags() -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(HOST_TAGS_URL, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return False, str(exc)
    except json.JSONDecodeError as exc:
        return False, str(exc)

    models = data.get("models") or []
    if not models:
        return True, "Nenhum modelo instalado (GET /api/tags retornou models: [])."

    lines = [str(m.get("name", "?")) for m in models]
    table = "NAME (via /api/tags)\n" + "\n".join(lines)
    return True, table


def main() -> int:
    model = resolve_ollama_model()
    print(f"1) Tentando `docker exec {OLLAMA_CONTAINER} ollama list`...\n")
    ok, out = _from_docker()
    if ok:
        print(out)
        print(
            "\n(Dica: isso e o Ollama dentro do container usado pelo Compose.)",
            file=sys.stderr,
        )
        return 0

    print(f"   Falhou: {out}\n")
    print(f"2) Tentando GET {HOST_TAGS_URL} (Ollama no host Windows)...\n")
    ok2, out2 = _from_host_tags()
    if ok2:
        print(out2)
        if "Nenhum modelo" in out2:
            print(
                f"\nInstale com: ollama pull {model}",
                file=sys.stderr,
            )
        return 0

    print(f"   Falhou: {out2}")
    print(
        "\nSuba o Ollama (ou `docker compose up -d ollama`) e tente de novo.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
