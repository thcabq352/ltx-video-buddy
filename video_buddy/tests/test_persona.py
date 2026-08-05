"""Tests for the persona system + intake interview (mocked LLM, no GPU).

Run: .venv/Scripts/python.exe -m pytest tests/test_persona.py -q
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from master_agent.persona.intake import (
    CreativeBrief,
    IntakeSession,
    run_interview_cli,
)
from master_agent.persona.persona import list_personas, load_persona


class _FakeResp:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    """Queues canned JSON replies regardless of input."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return _FakeResp(self._replies.pop(0) if self._replies else "{}")


class TestPersonaLoading(unittest.TestCase):
    def test_bundled_ara_and_exec_load(self):
        slugs = {p.slug for p in list_personas()}
        self.assertIn("ara", slugs)
        self.assertIn("exec", slugs)
        ara = load_persona("ara")
        self.assertEqual("Ara", ara.name)
        self.assertIn("creative director", ara.system_prompt)

    def test_default_is_ara(self):
        self.assertEqual("ara", load_persona().slug)

    def test_user_override_wins(self):
        with TemporaryDirectory() as td:
            override = Path(td) / "ara.md"
            override.write_text("# Ara\n\nCustom override persona.", encoding="utf-8")
            with patch("master_agent.persona.persona.PERSONA_DIR", Path(td)):
                ara = load_persona("ara")
                self.assertIn("Custom override persona.", ara.system_prompt)
                self.assertEqual(override, ara.path)

    def test_unknown_persona_lists_available(self):
        with self.assertRaises(ValueError) as ctx:
            load_persona("nobody")
        self.assertIn("ara", str(ctx.exception))


class TestCreativeBrief(unittest.TestCase):
    def test_to_request_includes_key_fields(self):
        brief = CreativeBrief(
            refined_request="chrome sneaker spot on wet asphalt",
            audience="sneakerheads 18-30",
            tone="kinetic, premium",
            style="neon night, macro textures",
        )
        req = brief.to_request()
        self.assertIn("chrome sneaker spot", req)
        self.assertIn("sneakerheads 18-30", req)
        self.assertIn("kinetic", req)

    def test_from_dict_tolerates_missing(self):
        brief = CreativeBrief.from_dict({"refined_request": "x"}, fallback_request="fb")
        self.assertEqual("x", brief.refined_request)
        self.assertEqual("", brief.audience)
        empty = CreativeBrief.from_dict({}, fallback_request="fb")
        self.assertEqual("fb", empty.refined_request)


_BRIEF_JSON = json.dumps(
    {
        "spoken": "Perfect, here's the brief.",
        "done": True,
        "brief": {
            "refined_request": "neon sneaker spot, rain-slick street",
            "audience": "sneakerheads",
            "tone": "kinetic",
        },
    }
)


class _IsolatedRunsDir(unittest.TestCase):
    """Redirect intake writes away from the real state/runs/ directory."""

    def setUp(self):
        self._runs_td = TemporaryDirectory()
        self._runs_dir = Path(self._runs_td.name)
        self._runs_patch = patch(
            "master_agent.persona.intake.RUNS_DIR", self._runs_dir
        )
        self._runs_patch.start()
        self.addCleanup(self._runs_patch.stop)
        self.addCleanup(self._runs_td.cleanup)


class TestIntakeSession(_IsolatedRunsDir):
    def _session(self, replies):
        session = IntakeSession("a sneaker commercial", persona=load_persona("ara"))
        fake = _FakeLLM(replies)
        session._llm = fake
        return session, fake

    def test_opening_uses_llm(self):
        session, fake = self._session([json.dumps({"spoken": "Hey! Who's it for?", "done": False})])
        self.assertEqual("Hey! Who's it for?", session.opening())
        self.assertEqual(1, fake.calls)

    def test_reply_parses_json_with_surrounding_prose(self):
        session, _ = self._session(["Sure thing!\n" + _BRIEF_JSON + "\nHope that helps"])
        rep = session.reply("for sneakerheads, kinetic")
        self.assertTrue(rep.done)
        self.assertIsNotNone(rep.brief)
        self.assertIn("neon sneaker spot", rep.brief.refined_request)

    def test_bail_phrase_skips_to_brief(self):
        session, fake = self._session([_BRIEF_JSON])
        rep = session.reply("just go")
        self.assertTrue(rep.done)
        self.assertIsNotNone(rep.brief)

    def test_max_rounds_auto_finalizes(self):
        session, _ = self._session(
            [json.dumps({"spoken": "more?", "done": False})] * 10
        )
        with patch("master_agent.persona.intake.INTAKE_MAX_ROUNDS", 2):
            rep = None
            for _i in range(3):
                rep = session.reply("some answer")
        self.assertTrue(rep.done)

    def test_no_llm_fallback_produces_request(self):
        session = IntakeSession("a sneaker commercial", persona=load_persona("ara"))
        session._llm_failed = True  # simulate provider down
        opening = session.opening()
        self.assertIn("who is this for", opening.lower())
        rep = session.reply("sneakerheads, neon, night city")
        self.assertFalse(rep.done)
        rep = session.reply("just go")
        self.assertTrue(rep.done)
        self.assertIn("sneakerheads", rep.brief.to_request())
        self.assertIn("sneaker commercial", rep.brief.to_request())

    def test_intake_record_written(self):
        session, _ = self._session([_BRIEF_JSON])
        session.reply("wrap it up please — just go")
        files = list(self._runs_dir.glob("*_intake.json"))
        self.assertEqual(1, len(files))
        rec = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual("intake", rec["kind"])
        self.assertEqual("ara", rec["persona"])
        self.assertIsNotNone(rec["brief"])


class TestCliLoop(_IsolatedRunsDir):
    def test_run_interview_cli_completes(self):
        session_replies = [_BRIEF_JSON]
        outputs = []
        with patch.object(IntakeSession, "_get_llm", lambda self: _FakeLLM(session_replies)):
            brief = run_interview_cli(
                "a sneaker commercial",
                input_fn=lambda _prompt: "just go",
                output_fn=outputs.append,
            )
        self.assertIn("neon sneaker spot", brief.refined_request)
        self.assertTrue(any("creative brief" in o for o in outputs))


if __name__ == "__main__":
    unittest.main()
