"""
projects.py
===========

THE single source of truth for "which projects exist and should be shown."

Both the frontend (via GET /projects) and the chatbot (via rag.py) call
get_visible_projects(). Because they read the same function, they can never
disagree about what's hidden. Fix a bug here, both surfaces get the fix.

Flow:
    live GitHub repos  +  your project_overrides.json  ->  final visible list
        (github_sync)          (hidden / pinned / overrides)

A hidden repo is removed HERE, before either surface sees it. That's what makes
"hidden" mean hidden *everywhere* rather than just hidden on the webpage.

Two robustness rules matter here:
  * Repo names on GitHub can be ANY case (e.g. "AI-Codebase-Tutor"). We match
    override / hidden / pinned keys case-INSENSITIVELY so you can write them in
    lowercase and never worry about exact casing.
  * All file paths are resolved relative to THIS file, not the current working
    directory, so it behaves the same whether you launch uvicorn from backend/
    or from the repo root or on Render.
"""

import os
import json
import time

import github_sync

# Resolve the data file relative to this file — NOT the working directory.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OVERRIDES_PATH = os.path.join(_BASE_DIR, "data", "project_overrides.json")

# Small in-memory cache so we don't call GitHub on every single page load.
# GitHub data changes slowly; 10 minutes is plenty fresh for a portfolio.
_CACHE = {"data": None, "ts": 0.0}
_CACHE_TTL_SECONDS = 600


def load_overrides() -> dict:
    """Read your control file. Missing file -> sensible empty defaults."""
    try:
        with open(OVERRIDES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
    return {
        "hidden": data.get("hidden", []),
        "pinned": data.get("pinned", []),
        "overrides": data.get("overrides", {}),
    }


def get_hidden_set() -> set:
    """
    Lowercased names of hidden repos, so callers can compare case-insensitively.
    rag.py / main.py use this as a live safety filter for the chatbot.
    """
    return {h.lower() for h in load_overrides()["hidden"]}


def _prettify(name: str) -> str:
    """Turn 'ai-codebase-tutor' into 'Ai Codebase Tutor' as a fallback title."""
    return name.replace("-", " ").replace("_", " ").title()


def _shape_card(repo: dict, override: dict) -> dict:
    """
    Build the final object the frontend renders. Your override fields win;
    anything you didn't specify falls back to the live GitHub info.
    """
    topics = repo.get("topics") or []
    lang = repo.get("language")
    # Default tech pills = language + a few topics, if you didn't set your own.
    default_tech = ([lang] if lang else []) + [t for t in topics if t != lang]

    return {
        "repo": repo["name"],
        "name": override.get("display_name") or _prettify(repo["name"]),
        "type": override.get("type")
        or (", ".join(topics[:3]) if topics else (lang or "Project")),
        "description": override.get("description")
        or repo.get("description")
        or "",
        "tech": override.get("tech") or default_tech[:6],
        "url": repo.get("url"),
        "homepage": repo.get("homepage") or None,
    }


def _sort_key(pinned_lower: list):
    """Pinned repos first (in your order), then everything else by recency."""
    def key(card):
        name = card["repo"].lower()
        return (pinned_lower.index(name) if name in pinned_lower else len(pinned_lower),)
    return key


def get_visible_projects(force_refresh: bool = False,
                         include_readme: bool = False) -> list[dict]:
    """
    Return the final, ready-to-render list of visible projects.

    Steps:
      1. fetch every repo from GitHub (cached for 10 min)
      2. drop the ones in your 'hidden' list      <- the privacy step
      3. merge each remaining repo with your override copy
      4. order them: pinned first, then most-recently-pushed
    """
    now = time.time()
    if (not force_refresh
            and _CACHE["data"] is not None
            and now - _CACHE["ts"] < _CACHE_TTL_SECONDS):
        repos = _CACHE["data"]
    else:
        repos = github_sync.fetch_repos(include_readme=include_readme)
        # Only cache a real (non-empty) result. If GitHub failed and returned
        # [], keep any older good cache instead of blanking the page.
        if repos:
            _CACHE["data"] = repos
            _CACHE["ts"] = now
        elif _CACHE["data"] is not None:
            repos = _CACHE["data"]

    ov = load_overrides()

    # Build case-INSENSITIVE lookups so "portfolio-ai" in your file matches a
    # repo GitHub returns as "Portfolio-AI", "portfolio-ai", etc.
    overrides_lower = {k.lower(): v for k, v in ov["overrides"].items()}
    hidden_lower = {h.lower() for h in ov["hidden"]}
    pinned_lower = [p.lower() for p in ov["pinned"]]

    cards = [
        _shape_card(r, overrides_lower.get(r["name"].lower(), {}))
        for r in repos
        # hidden removed here
        if r.get("name") and r["name"].lower() not in hidden_lower
    ]

    cards.sort(key=_sort_key(pinned_lower))
    return cards


def invalidate_cache():
    """Call this right after editing the hidden list so changes show instantly."""
    _CACHE["data"] = None
    _CACHE["ts"] = 0.0
