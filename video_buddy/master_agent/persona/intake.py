"""Intake interview engine — shared by the CLI and the web UI.

The persona interviews the user (LLM-driven, JSON replies) until the picture
is complete, then hands over a structured CreativeBrief that becomes the
pipeline request. Degrades gracefully: with no LLM available the session
still works — canned opener, answers folded into the request.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from master_agent.config import INTAKE_MAX_ROUNDS, RUNS_DIR
from master_agent.persona.persona import Persona, load_persona

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "intake.md"

_BAIL_PHRASES = (
    "just go",
    "skip",
    "go ahead",
    "generate",
    "go for it",
    "that's enough",
    "thats enough",
    "let's go",
    "lets go",
)

_FALLBACK_OPENING = (
    "Love it. Before I spin anything up — who is this for, what tone should "
    "it strike, and are there any must-haves or deal-breakers?"
)


@dataclass
class CreativeBrief:
    refined_request: str
    audience: str = ""
    tone: str = ""
    style: str = ""
    constraints: str = ""
    duration_s: Optional[float] = None
    aspect_ratio: str = ""
    music_mood: Optional[str] = None
    negatives: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any], fallback_request: str = "") -> "CreativeBrief":
        return cls(
            refined_request=str(d.get("refined_request") or fallback_request),
            audience=str(d.get("audience") or ""),
            tone=str(d.get("tone") or ""),
            style=str(d.get("style") or ""),
            constraints=str(d.get("constraints") or ""),
            duration_s=d.get("duration_s") if isinstance(d.get("duration_s"), (int, float)) else None,
            aspect_ratio=str(d.get("aspect_ratio") or ""),
            music_mood=d.get("music_mood") or None,
            negatives=str(d.get("negatives") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_request(self) -> str:
        """Render the enriched pipeline request."""
        parts = [self.refined_request.strip()]
        extras = []
        if self.audience:
            extras.append(f"Audience: {self.audience}")
        if self.tone:
            extras.append(f"Tone: {self.tone}")
        if self.style:
            extras.append(f"Style: {self.style}")
        if self.constraints:
            extras.append(f"Constraints: {self.constraints}")
        if extras:
            parts.append(" — ".join(extras))
        return "\n".join(p for p in parts if p)


@dataclass
class IntakeReply:
    spoken: str
    done: bool = False
    brief: Optional[CreativeBrief] = None


class IntakeSession:
    """One interview. CLI drives it with input(); the web UI via /api/intake."""

    def __init__(
        self,
        initial_request: str,
        *,
        persona: Optional[Persona] = None,
        provider: Optional[str] = None,
    ):
        self.id = uuid.uuid4().hex[:12]
        self.initial_request = initial_request.strip()
        self.persona = persona or load_persona()
        self.provider = provider
        self.history: list[dict[str, str]] = []
        self.rounds = 0
        self._brief: Optional[CreativeBrief] = None
        self._last_spoken: Optional[str] = None
        self._llm = None
        self._llm_failed = False

    # ── LLM plumbing ──────────────────────────────────────

    def _system(self) -> str:
        template = (
            _PROMPT_PATH.read_text(encoding="utf-8")
            if _PROMPT_PATH.is_file()
            else "{persona}\n\n{soul}\nInterview the user. Reply JSON only."
        )
        from master_agent.persona.soul import load_soul

        soul = load_soul()
        return (
            template.replace("{persona}", self.persona.system_prompt)
            .replace("{soul}", soul.system_prompt)
        )

    def _get_llm(self):
        if self._llm is None and not self._llm_failed:
            try:
                from master_agent.llm import get_llm

                self._llm = get_llm(temperature=0.4, provider=self.provider)
            except Exception:
                self._llm_failed = True
        return self._llm

    def _invoke(self, user_text: str) -> Optional[dict[str, Any]]:
        """One LLM round; returns parsed JSON or None (fallback)."""
        llm = self._get_llm()
        if llm is None:
            return None
        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        except Exception:
            return None
        messages: list = [SystemMessage(content=self._system())]
        if not self.history:
            user_text = f"The user's initial idea: {self.initial_request}\n\n{user_text}"
        else:
            for turn in self.history:
                cls = HumanMessage if turn["role"] == "user" else AIMessage
                messages.append(cls(content=turn["text"]))
        messages.append(HumanMessage(content=user_text))
        try:
            resp = llm.invoke(messages)
            return _extract_json(getattr(resp, "content", None) or str(resp))
        except Exception:
            self._llm_failed = True
            return None

    # ── conversation ──────────────────────────────────────

    def opening(self) -> str:
        """Her first message: greet + first questions."""
        data = self._invoke("Start the interview.")
        if data and data.get("spoken"):
            spoken = str(data["spoken"])
        else:
            spoken = _FALLBACK_OPENING
        self.history.append({"role": "agent", "text": spoken})
        return spoken

    def reply(self, user_text: str) -> IntakeReply:
        """Process one user message -> her reply (+ brief when done)."""
        user_text = (user_text or "").strip()
        if not user_text:
            return IntakeReply(spoken="I'm listening — what's on your mind?")
        self.history.append({"role": "user", "text": user_text})
        self.rounds += 1

        if self._wants_out(user_text) or self.rounds > INTAKE_MAX_ROUNDS:
            brief = self.finalize()
            spoken = self._last_spoken or "Perfect — I have everything I need. Here's the brief."
            return IntakeReply(spoken=spoken, done=True, brief=brief)

        data = self._invoke(user_text)
        if data is None:
            # No LLM: keep collecting; bail-out/max-rounds handled above
            spoken = "Noted — anything else I should know? (say 'just go' when ready)"
            self.history.append({"role": "agent", "text": spoken})
            return IntakeReply(spoken=spoken)

        spoken = str(data.get("spoken") or "…")
        self.history.append({"role": "agent", "text": spoken})
        if data.get("done"):
            brief = self._brief_from(data.get("brief"))
            self._write_record()
            return IntakeReply(spoken=spoken, done=True, brief=brief)
        return IntakeReply(spoken=spoken)

    def finalize(self) -> CreativeBrief:
        """Close the interview now — LLM brief if possible, heuristic otherwise."""
        if self._brief is not None:
            return self._brief
        data = self._invoke(
            "The user wants to proceed now. Close out: done=true with the full brief."
        )
        if data and data.get("spoken"):
            self._last_spoken = str(data["spoken"])
        brief = self._brief_from((data or {}).get("brief"))
        self._write_record()
        return brief

    # ── helpers ───────────────────────────────────────────

    def _brief_from(self, raw: Any) -> CreativeBrief:
        if isinstance(raw, dict) and raw.get("refined_request"):
            brief = CreativeBrief.from_dict(raw, fallback_request=self.initial_request)
        else:
            brief = self._fallback_brief()
        self._brief = brief
        return brief

    def _fallback_brief(self) -> CreativeBrief:
        answers = [t["text"] for t in self.history if t["role"] == "user"]
        refined = self.initial_request
        if answers:
            refined = f"{self.initial_request} — {'; '.join(answers)}"
        return CreativeBrief(refined_request=refined)

    @staticmethod
    def _wants_out(text: str) -> bool:
        # Normalize punctuation so "wrap it up — just go" still bails
        low = re.sub(r"[,;:–—\-]+", " ", (text or "").lower())
        low = re.sub(r"\s+", " ", low).strip().rstrip(".!")
        return any(
            low == p
            or low.startswith(p + " ")
            or low.endswith(" " + p)
            or f" {p} " in f" {low} "
            for p in _BAIL_PHRASES
        )

    def _write_record(self) -> None:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        record = {
            "run_id": self.id,
            "kind": "intake",
            "persona": self.persona.slug,
            "initial_request": self.initial_request,
            "history": self.history,
            "brief": self._brief.to_dict() if self._brief else None,
        }
        try:
            (RUNS_DIR / f"{ts}_{self.id}_intake.json").write_text(
                json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8"
            )
        except OSError:
            pass


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def run_interview_cli(
    initial_request: str,
    *,
    persona: Optional[Persona] = None,
    provider: Optional[str] = None,
    input_fn=input,
    output_fn=print,
) -> CreativeBrief:
    """Terminal interview loop; returns the creative brief."""
    session = IntakeSession(initial_request, persona=persona, provider=provider)
    name = session.persona.name
    output_fn(f"\n{name} · creative director")
    output_fn(f"{name}: {session.opening()}")
    while True:
        try:
            answer = input_fn("you: ")
        except (EOFError, KeyboardInterrupt):
            answer = "just go"
        rep = session.reply(answer)
        output_fn(f"{name}: {rep.spoken}")
        if rep.done and rep.brief:
            output_fn("\n--- creative brief ---")
            output_fn(rep.brief.to_request())
            output_fn("----------------------\n")
            return rep.brief
