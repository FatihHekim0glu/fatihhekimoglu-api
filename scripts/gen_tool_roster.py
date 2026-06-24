#!/usr/bin/env python3
"""Derive the README tool roster from ``api/main.py`` so it can't drift.

The single source of truth for which tools are served — and in what order — is
the sequence of ``app.include_router(<module>.router)`` calls in
``api/main.py``. This script parses those calls statically (no app boot, no
network), imports each referenced router module to read its ``prefix`` (the tool
slug) and router file path, and regenerates the markdown table in ``README.md``.

Usage
-----
    python scripts/gen_tool_roster.py            # rewrite README.md in place
    python scripts/gen_tool_roster.py --check    # exit 1 if README is stale
    python scripts/gen_tool_roster.py --print     # print the table to stdout

The table is written between the ``<!-- TOOL-ROSTER:START -->`` and
``<!-- TOOL-ROSTER:END -->`` markers in README.md. Keep those markers in place.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MAIN_PY = _REPO_ROOT / "api" / "main.py"
_README = _REPO_ROOT / "README.md"

_START_MARKER = "<!-- TOOL-ROSTER:START -->"
_END_MARKER = "<!-- TOOL-ROSTER:END -->"


def _included_router_modules(main_py: Path) -> list[str]:
    """Return the router module names in ``include_router`` order.

    Statically parses ``app.include_router(<module>.router)`` calls so the
    roster order matches the served order without booting the app.
    """
    tree = ast.parse(main_py.read_text(encoding="utf-8"), filename=str(main_py))
    modules: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Match `<something>.include_router(...)`.
        if not (isinstance(func, ast.Attribute) and func.attr == "include_router"):
            continue
        if not node.args:
            continue
        arg = node.args[0]
        # Match `<module>.router`.
        if isinstance(arg, ast.Attribute) and arg.attr == "router" and isinstance(
            arg.value, ast.Name
        ):
            modules.append(arg.value.id)
    return modules


def _roster_rows(modules: list[str]) -> list[tuple[str, str, str]]:
    """Return ``(slug, router_path, openapi_tag)`` rows for each router."""
    rows: list[tuple[str, str, str]] = []
    for module_name in modules:
        module = importlib.import_module(f"api.routers.{module_name}")
        router = module.router
        slug = router.prefix.removeprefix("/tools/") or router.prefix
        tag = router.tags[0] if router.tags else slug
        router_path = f"api/routers/{module_name}.py"
        rows.append((slug, router_path, str(tag)))
    return rows


def _render_table(rows: list[tuple[str, str, str]]) -> str:
    lines = [
        f"_{len(rows)} tools served. Auto-generated from `api/main.py` by "
        "`scripts/gen_tool_roster.py` — do not edit by hand._",
        "",
        "| Slug | Router file | OpenAPI tag |",
        "|---|---|---|",
    ]
    for slug, path, tag in rows:
        lines.append(f"| `{slug}` | `{path}` | `{tag}` |")
    return "\n".join(lines)


def _splice_readme(readme_text: str, table: str) -> str:
    if _START_MARKER not in readme_text or _END_MARKER not in readme_text:
        raise SystemExit(
            f"README is missing the {_START_MARKER} / {_END_MARKER} markers; "
            "add them around the tool-roster table first."
        )
    head, rest = readme_text.split(_START_MARKER, 1)
    _, tail = rest.split(_END_MARKER, 1)
    return f"{head}{_START_MARKER}\n{table}\n{_END_MARKER}{tail}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if README is stale")
    parser.add_argument("--print", action="store_true", help="print the table only")
    args = parser.parse_args(argv)

    modules = _included_router_modules(_MAIN_PY)
    rows = _roster_rows(modules)
    table = _render_table(rows)

    if args.print:
        print(table)
        return 0

    current = _README.read_text(encoding="utf-8")
    updated = _splice_readme(current, table)

    if args.check:
        if current != updated:
            print("README tool roster is stale; run scripts/gen_tool_roster.py", file=sys.stderr)
            return 1
        print("README tool roster is up to date.")
        return 0

    if current != updated:
        _README.write_text(updated, encoding="utf-8")
        print(f"README tool roster updated ({len(rows)} tools).")
    else:
        print("README tool roster already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
