"""
github_sync.py
==============

Talks to the GitHub REST API and returns a clean list of "projects" built from
your public repositories.

Design notes:
- Uses only the Python standard library (urllib) so there's nothing new to
  install. This matches the approach already used in the Codebase-Tutor repo.
- A GITHUB_TOKEN is OPTIONAL. Without one, GitHub allows ~60 requests/hour per
  IP (fine for a small portfolio). With one, you get 5000/hour and can also see
  private repos. Put it in your .env as GITHUB_TOKEN=ghp_xxx if you want that.
- This file does NOT decide what is hidden. It just reports what's on GitHub.
  Hiding/curation happens one layer up, in projects.py. Keeping "what exists"
  and "what to show" separate is what keeps the whole thing easy to reason about.
"""

import os
import json
import base64
import urllib.request
import urllib.error

GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "tryadityagupta")


def _get(url: str):
    """A tiny GET helper. Returns parsed JSON, or None on any failure."""
    token = os.getenv("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "portfolio-ai-sync",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())
    except Exception:
        # Network hiccup, rate limit, 404, etc. We fail soft: the caller will
        # fall back to whatever it had before rather than crashing the site.
        return None


def fetch_readme(repo_name: str) -> str:
    """
    Grab a repo's README text so the chatbot has richer context than the
    one-line description. Returns "" if the repo has no README.
    """
    data = _get(
        f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/readme")
    if not data or "content" not in data:
        return ""
    try:
        # GitHub returns the README base64-encoded.
        text = base64.b64decode(data["content"]).decode(
            "utf-8", errors="ignore")
        # Keep it reasonable so one huge README doesn't dominate the index.
        return text[:4000]
    except Exception:
        return ""


def fetch_repos(include_readme: bool = False) -> list[dict]:
    """
    Return a normalized list of your repos, newest activity first.

    Each item looks like:
        {
          "name": "email-classifier",
          "description": "...",          # GitHub's description (may be None)
          "url": "https://github.com/...",
          "homepage": "https://...",     # if you set a website on the repo
          "language": "Python",
          "topics": ["nlp", "fastapi"],  # the repo's GitHub topics/tags
          "pushed_at": "2026-06-30T...",
          "readme": "..."                # only if include_readme=True
        }

    Forks and archived repos are dropped by default — a portfolio should show
    your own, current work. (You can still un-hide any repo explicitly.)
    """
    raw = _get(
        f"https://api.github.com/users/{GITHUB_USERNAME}/repos"
        f"?per_page=100&sort=updated&type=owner"
    )
    if not raw:
        return []

    projects = []
    for r in raw:
        if r.get("fork") or r.get("archived"):
            continue
        item = {
            "name": r.get("name"),
            "description": r.get("description"),
            "url": r.get("html_url"),
            "homepage": r.get("homepage"),
            "language": r.get("language"),
            "topics": r.get("topics", []) or [],
            "pushed_at": r.get("pushed_at"),
        }
        if include_readme:
            item["readme"] = fetch_readme(item["name"])
        projects.append(item)

    return projects


def fetch_commits(repo_name: str, max_commits: int = 200) -> list[dict]:
    """
    Newest-first commit history for one repo:

        [{"date": "2026-08-19", "message": "feat: add semantic cache"}, ...]

    Only the first line of each message (git's subject-line convention) and
    the author date, day precision — that's all the chatbot needs and it
    keeps the embedded documents small. Pages through the API at 100 per
    call until max_commits or the history ends; every failure fails SOFT to
    whatever was collected so far (possibly []), same contract as the rest
    of this file. Merge commits are kept: "Merge pull request #1 ..." is a
    real, informative event in a project's story.
    """
    commits: list[dict] = []
    page, per_page = 1, 100
    while len(commits) < max_commits:
        data = _get(
            f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}"
            f"/commits?per_page={per_page}&page={page}"
        )
        if not data or not isinstance(data, list):
            break
        for c in data:
            info = c.get("commit") or {}
            message = (info.get("message") or "").strip().split("\n")[0][:200]
            date = ((info.get("author") or {}).get("date") or "")[:10]
            if message:
                commits.append({"date": date, "message": message})
            if len(commits) >= max_commits:
                break
        if len(data) < per_page:
            break
        page += 1
    return commits
