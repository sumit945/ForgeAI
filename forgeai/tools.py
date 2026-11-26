"""Tool registry. Each tool declares the *evidence* that must exist in the brief/requirements for it to be attached.

The generator never trusts the LLM alone: a tool is attached only if (a) the LLM votes for it AND
(b) deterministic evidence exists (regex hits in the requirements and, for csv_lookup, a CSV in the folder).
"""
from __future__ import annotations

import ast
import datetime as dt
import operator
import re
from dataclasses import dataclass
from typing import Callable

import requests


@dataclass
class Tool:
    name: str
    description: str
    args: dict
    evidence: list[str]  # regexes; at least one must match the brief+requirements text
    needs_csv: bool = False
    fn: Callable | None = None


# ---- calculator -----------------------------------------------------------------------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCS = {"round": round, "min": min, "max": max, "abs": abs, "sum": lambda *a: sum(a)}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        l, r = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(r) > 12:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](l, r)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("unsupported expression")


def calculator(args: dict, ctx: dict) -> dict:
    expr = str(args.get("expression", "")).replace(",", "").replace("%", "/100").replace("^", "**")
    try:
        val = _eval(ast.parse(expr, mode="eval"))
        return {"expression": args.get("expression"), "result": round(val, 6) if isinstance(val, float) else val}
    except Exception as e:  # noqa: BLE001
        return {"error": f"cannot evaluate: {e}"}


# ---- current date ---------------------------------------------------------------------
def current_date(args: dict, ctx: dict) -> dict:
    now = dt.datetime.now()
    out = {"today": now.date().isoformat(), "weekday": now.strftime("%A"), "time": now.strftime("%H:%M")}
    other = args.get("other_date")
    if other:
        try:
            d = dt.date.fromisoformat(str(other))
            out["days_until_other_date"] = (d - now.date()).days
        except ValueError:
            out["error"] = "other_date must be YYYY-MM-DD"
    return out


# ---- csv lookup -----------------------------------------------------------------------
def csv_lookup(args: dict, ctx: dict) -> dict:
    from .retrieval import tokenize

    docs = [d for d in ctx.get("documents", []) if d.kind == "csv"]
    if not docs:
        return {"error": "no csv available"}
    fname = args.get("file")
    if fname:
        docs = [d for d in docs if d.name == fname] or docs
    query = str(args.get("query", ""))
    col, eq = args.get("column"), args.get("equals")
    qt = set(tokenize(query))
    hits = []
    for d in docs:
        for r in d.rows:
            if col and eq is not None:
                if str(r.get(col, "")).strip().lower() != str(eq).strip().lower():
                    continue
                hits.append((9.0, d.name, r))
                continue
            rt = set(tokenize(" ".join(r.values())))
            s = len(qt & rt)
            if s:
                hits.append((s / max(1, len(qt)), d.name, r))
    hits.sort(key=lambda h: -h[0])
    return {"matches": [{"file": f, "row": r} for _, f, r in hits[:5]], "count": len(hits)}


# ---- open-meteo -----------------------------------------------------------------------
def open_meteo(args: dict, ctx: dict) -> dict:
    city = str(args.get("city", "")).strip()
    if not city:
        return {"error": "city required"}
    try:
        g = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 1}, timeout=15).json()
        if not g.get("results"):
            return {"error": f"city not found: {city}"}
        loc = g["results"][0]
        days = max(1, min(int(args.get("days", 3)), 7))
        w = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": loc["latitude"], "longitude": loc["longitude"], "timezone": "auto", "forecast_days": days,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
            },
            timeout=15,
        ).json()
        return {"location": f"{loc['name']}, {loc.get('country', '')}", "daily": w.get("daily", {})}
    except Exception as e:  # noqa: BLE001
        return {"error": f"weather service unavailable: {type(e).__name__}"}


REGISTRY: dict[str, Tool] = {
    "csv_lookup": Tool(
        "csv_lookup",
        "Look up rows in a CSV table of the knowledge base (part numbers, prices, stock, etc.).",
        {"query": "words to match against the row", "file": "(optional) csv file name", "column": "(optional)", "equals": "(optional) exact value for column"},
        [r"\b(catalog(ue)?|inventory|stock|part number|spare|price list|sku|lookup|look up|table|register|roster|schedule of)\b"],
        needs_csv=True, fn=csv_lookup,
    ),
    "calculator": Tool(
        "calculator",
        "Evaluate an arithmetic expression exactly (use instead of mental maths).",
        {"expression": "e.g. 12 * 4.5 + 3"},
        [r"\b(calculat\w*|comput\w*|totals?|sum|quote[sd]?|budget\w*|convert\w*|percent\w*|accru\w*|pro[- ]?rata|estimat\w*|multipl\w*|order value|unit price|line cost)\b"],
        fn=calculator,
    ),
    "open_meteo_weather": Tool(
        "open_meteo_weather",
        "Get a short public weather forecast for a city (Open-Meteo, no API key).",
        {"city": "city or port name", "days": "1-7"},
        [r"\b(weather|forecast|temperature|rain\w*|wind|sea state|storm|climate|met(eorolog\w*)?)\b"],
        fn=open_meteo,
    ),
    "current_date": Tool(
        "current_date",
        "Get today's date/time; optionally days until another date (YYYY-MM-DD).",
        {"other_date": "(optional) YYYY-MM-DD"},
        [r"\b(today|current date|deadline|due (date|by)|expir\w*|how (long|many days)|upcoming|days (left|until|since)|renewal|valid until|notice period)\b"],
        fn=current_date,
    ),
}


def run_tool(name: str, args: dict, ctx: dict) -> dict:
    t = REGISTRY.get(name)
    if not t or not t.fn:
        return {"error": f"unknown tool {name}"}
    try:
        return t.fn(args if isinstance(args, dict) else {}, ctx)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def evidence_hits(name: str, text: str) -> list[str]:
    t = REGISTRY[name]
    hits: list[str] = []
    for pat in t.evidence:
        hits += [m.group(0).lower() for m in re.finditer(pat, text, flags=re.I)]
    return sorted(set(hits))
