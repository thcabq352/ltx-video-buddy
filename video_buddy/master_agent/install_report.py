"""Shared wording for the end of ``install.py`` / ``setup --fix`` / automatic install.

Every line a new user sees is one of four levels:

- ``OK`` — done.
- ``INFO`` — expected for this kind of install; nothing to do now.
- ``FAILED-NONFATAL`` — something did not work, but the install is still usable.
- ``FAIL`` — the install is not usable yet; the command exits non-zero.

``INFO`` and ``FAILED-NONFATAL`` always carry a one-line "safe to ignore because ...".
Only ``FAIL`` counts against the exit code. All text is ASCII so Windows
consoles on legacy code pages print it as-is.
"""

from __future__ import annotations

OK = "OK"
INFO = "INFO"
NONFATAL = "FAILED-NONFATAL"
FAIL = "FAIL"

FULL_INSTALL_CMD = "python install.py --automatic-install --yes"
FULL_INSTALL_CMD_WINDOWS = "install.bat --automatic-install --yes"

# Doctor rows that the light install (plain install.py / setup --fix) never
# provides, by design. After a light install they are INFO, not failures.
LIGHT_INSTALL_NOT_INCLUDED: dict[str, str] = {
    "comfyui": "the light install does not install or start ComfyUI (the full setup does)",
    "ltx23-weights": f"the light install downloads no weights; {FULL_INSTALL_CMD} fetches LTX 2.3",
    "ltx25-weights": "LTX 2.5 is an optional gated pack; only the ltx25_* workflows need it",
    "ltx25-ic-lora": "only ltx25_msr and ltx25_v2v_ic_lora need this LoRA",
    "ltx23-inoutpaint-lora": "only ltx23_inoutpaint and ltx25_inoutpaint need this LoRA",
    "ltx23-upscaler": "only lipsync, ltx23_i2v and vb_movie_builder need this upscaler",
    "h3-weights": "MiniMax H3 is an optional pack; only the h3_* workflows need it",
    "dev-fp8-trailer": "it only checks a dev checkpoint that is not part of any install",
}

# Report-only doctor rows. They never fail doctor, light install or not.
INFO_ROW_REASONS: dict[str, str] = {
    "llm-models": "no public GGUF source is defined for them; Buddy renders video without a local LLM",
    "heartlib": "HeartMuLa music is optional; video generation does not use it",
    "heartmula-weights": "HeartMuLa music is optional; video generation does not use it",
    "heartmula-comfy": "HeartMuLa music is optional; video generation does not use it",
    "hardware": "it is information only and never blocks install",
    "pack-pins": "nothing was updated; stale pins only matter for an installed ComfyUI",
    "model-folders": "each download creates the folder it writes into",
}

# Optional automatic-install steps that may end FAILED-NONFATAL / NEEDS-YOU.
STEP_REASONS: dict[str, str] = {
    ".env": "Buddy runs on its built-in defaults without .env",
    "Playwright Chromium": "only the web scraper uses Chromium; video generation does not",
    "Triton + SageAttention": "ComfyUI runs without them, just slower",
    "llama.cpp": "Buddy renders video without a local LLM; the brief / judge LLM stays off",
    "LLM models": "no public GGUF source is defined; video generation does not need them",
}


def safe_to_ignore(why: str) -> str:
    return f"safe to ignore because {why}"


def summary_line(*, ok: int, notes: int, failed: list[str]) -> str:
    if failed:
        return (
            f"Install FAILED: {len(failed)} problem(s) to fix ({', '.join(failed)}); "
            f"{ok} ok, {notes} non-fatal note(s)."
        )
    return f"Install OK: {ok} ok, {notes} non-fatal note(s)."


def light_install_banner(*, where: str) -> str:
    """Loud ASCII box for plain ``install.py``. ``where`` is ``start`` or ``end``."""
    bar = "#" * 72
    if where == "start":
        state = "(starting)"
        skipped = "#  It will NOT install ComfyUI and will NOT download the LTX 2.3 weights,"
    else:
        state = "(finished)"
        skipped = "#  It did NOT install ComfyUI and did NOT download the LTX 2.3 weights,"
    return "\n".join(
        [
            bar,
            f"#  THIS IS THE LIGHT INSTALL {state}",
            "#",
            "#  It sets up .venv, Python packages, Playwright, .env, ffmpeg and the",
            "#  empty model folders only.",
            skipped,
            "#  so nothing can render and ComfyUI will not answer on :8188.",
            "#",
            "#  For the full setup (ComfyUI + LTX 2.3 + llama.cpp), run:",
            f"#      {FULL_INSTALL_CMD}",
            f"#  Windows:  {FULL_INSTALL_CMD_WINDOWS}",
            "#  Drop --yes to see the pre-flight and the plan without changing anything.",
            bar,
        ]
    )
