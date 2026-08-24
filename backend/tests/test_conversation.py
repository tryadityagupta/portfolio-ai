"""
Conversation-layer tests: session memory, question condensation, how
follow-ups interact with BOTH cache levels, the cross-script language
guard, and the exhaustive project roster.

The invariants worth paying for:

  * a session's FIRST turn must behave exactly like the pre-memory app
    (no condense call, exact cache consulted on the raw message)
  * a follow-up is condensed ONCE, and the standalone rewrite — not the
    raw follow-up — is what hits caches, embeddings and retrieval
  * a condensed follow-up may legitimately be served from either cache
    level (that's a feature: context-free keys by construction)
  * history is replayed to the model as real user/assistant turns
  * a cross-script semantic nomination is refused even above threshold
  * every prompt carries the COMPLETE PROJECT LIST, recency-ordered
  * if the condense call fails we still answer (with history) but write
    NOTHING to the caches — a raw follow-up must never become a key
"""

import pytest
from sqlalchemy import select

import analytics
from conftest import (
    Q_BASE, Q_HINDI, Q_OTHER, Q_PARA,
    chat, drain_writes,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def _turn_rows(question: str):
    async with analytics._Session() as s:
        return (await s.execute(
            select(analytics.ChatTurn)
            .where(analytics.ChatTurn.question == question)
        )).scalars().all()


async def _cached_rows():
    async with analytics._Session() as s:
        return (await s.execute(
            select(analytics.CachedAnswer))).scalars().all()


# ---------------------------------------------------------------------------
# 1. First turn of a session == the old single-turn app.
# ---------------------------------------------------------------------------

async def test_first_turn_has_no_history_and_no_condense(env):
    answer = await chat(env, Q_BASE, session_id="s1")
    assert answer == env.gpt.answer
    assert env.gpt.condense_calls == 0     # nothing to condense yet
    assert env.gpt.calls == 1              # one real generation
    assert env.embeddings.query_calls == 1


# ---------------------------------------------------------------------------
# 2. Follow-up -> condensed -> SEMANTIC hit on the standalone form.
# ---------------------------------------------------------------------------

async def test_followup_condenses_then_hits_semantic_cache(env):
    await chat(env, Q_BASE, session_id="s1")          # miss -> cached

    follow_up = "tell me more about that"
    env.gpt.condense_map[follow_up] = Q_PARA          # cos 0.95 to Q_BASE
    answer = await chat(env, follow_up, session_id="s1")

    assert answer == env.gpt.answer                   # replayed, not re-made
    assert env.gpt.condense_calls == 1                # exactly one rewrite
    assert env.gpt.calls == 1                         # STILL one generation
    # Embedded the standalone form, never the raw follow-up:
    assert env.embeddings.queries == [Q_BASE, Q_PARA]

    (row,) = await _turn_rows(follow_up)
    assert row.status == "semantic_hit"
    assert row.cache_similarity == pytest.approx(0.95, abs=1e-4)


# ---------------------------------------------------------------------------
# 3. Follow-up whose rewrite is an EXACT cached question -> free hit.
# ---------------------------------------------------------------------------

async def test_followup_condensed_to_exact_cache_hit(env):
    await chat(env, Q_BASE, session_id="s1")

    follow_up = "please repeat that"
    env.gpt.condense_map[follow_up] = Q_BASE
    answer = await chat(env, follow_up, session_id="s1")

    assert answer == env.gpt.answer
    assert env.gpt.condense_calls == 1
    assert env.gpt.calls == 1
    assert env.embeddings.query_calls == 1   # exact hit costs 0 embeddings

    (row,) = await _turn_rows(follow_up)
    assert row.status == "cache_hit"


# ---------------------------------------------------------------------------
# 4. History is replayed to the model as real conversation turns.
# ---------------------------------------------------------------------------

async def test_history_is_replayed_as_conversation_turns(env):
    await chat(env, Q_BASE, session_id="s1")
    await chat(env, Q_OTHER, session_id="s1")   # orthogonal -> real miss

    messages = env.gpt.prompts[-1]              # second generation's messages
    roles = [m["role"] for m in messages]
    assert roles[0] == "system"
    assert "assistant" in roles                 # the previous answer is there

    prev_user = next(m for m in messages if m["role"] == "user")
    assert prev_user["content"] == Q_BASE       # first question, verbatim
    prev_assistant = next(m for m in messages if m["role"] == "assistant")
    assert prev_assistant["content"].startswith(env.gpt.answer[:60])
    # The final user message is the RAG prompt built on the current question.
    assert Q_OTHER in messages[-1]["content"]


# ---------------------------------------------------------------------------
# 5. Language guard: same meaning, different script -> NOT served from cache.
# ---------------------------------------------------------------------------

async def test_cross_script_semantic_nomination_is_refused(env):
    await chat(env, Q_BASE)                     # seed an ENGLISH answer

    answer = await chat(env, Q_HINDI)           # cos 0.95 — above threshold
    assert answer == env.gpt.answer
    assert env.gpt.calls == 2                   # generated fresh, NOT replayed

    (row,) = await _turn_rows(Q_HINDI)
    assert row.status == "ok"                   # a miss, by design
    # ...but the near-miss similarity is still logged as tuning evidence:
    assert row.cache_similarity == pytest.approx(0.95, abs=1e-4)


# ---------------------------------------------------------------------------
# 6. Every prompt carries the exhaustive, recency-ordered project roster.
# ---------------------------------------------------------------------------

async def test_roster_reaches_model_exhaustive_and_recency_ordered(env):
    await chat(env, "List down all of his projects")

    final_prompt = env.gpt.prompts[-1][-1]["content"]
    assert "COMPLETE PROJECT LIST" in final_prompt
    assert "Portfolio AI Chatbot" in final_prompt
    assert "CareRoute" in final_prompt
    # portfolio-ai was pushed 2026-08-21, careroute 2026-07-01 -> newer first.
    assert (final_prompt.index("Portfolio AI Chatbot")
            < final_prompt.index("CareRoute"))
    assert "2026-08-21" in final_prompt         # push dates surfaced


# ---------------------------------------------------------------------------
# 7. Condense failure: still answers (with history), writes NO cache entry.
# ---------------------------------------------------------------------------

async def test_condense_failure_answers_but_skips_caches(env):
    await chat(env, Q_BASE, session_id="s1")
    assert len(await _cached_rows()) == 1       # turn 1 cached normally

    env.gpt.condense_error = True
    follow_up = "and what about the other one?"
    answer = await chat(env, follow_up, session_id="s1")
    await drain_writes()

    assert answer == env.gpt.answer             # degraded, not broken
    assert env.gpt.calls == 2                   # real generation happened
    # The raw follow-up must NOT have become a cache key:
    rows = await _cached_rows()
    assert len(rows) == 1
    assert rows[0].question_raw == Q_BASE

    (row,) = await _turn_rows(follow_up)
    assert row.status == "ok"
