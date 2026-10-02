# Contributing

Video Buddy docs have one home per fact.

| If you are changing… | Edit |
|---|---|
| Install, L0→L5 | [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) |
| How the agent is wired | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| A CLI command | [`docs/CLI_REFERENCE.md`](docs/CLI_REFERENCE.md), and `python -m master_agent <cmd> --help` |
| Weights, consent, loader order | [`docs/WEIGHTS.md`](docs/WEIGHTS.md) only |
| Comfy process or attach | [`docs/COMFY.md`](docs/COMFY.md) |
| Judge or the revise loop | [`docs/JUDGE.md`](docs/JUDGE.md) |
| Sidecar schema | [`docs/PROVENANCE.md`](docs/PROVENANCE.md) |
| A product feature | [`docs/FEATURES.md`](docs/FEATURES.md) |
| A stop-line | [`docs/AGENTS.md`](docs/AGENTS.md) |
| Wired / retired | [`docs/AUDIT.md`](docs/AUDIT.md) |

Index: [`docs/INDEX.md`](docs/INDEX.md).

Do not paste the loader order, the consent rules, or the pack table into a second file. Link to [Weights](docs/WEIGHTS.md#loader-policy). A second copy of that section means the dedup failed.

`docs/archive/` is the pre-slim manuals. Do not treat it as current. Do not delete it in a drive-by cleanup.

Retired capabilities stay retired. If [Audit](docs/AUDIT.md) says retired, do not describe it as coming soon.

`knowledge/` is memory between machines, not the manual. Follow [`knowledge/AGENTS.md`](knowledge/AGENTS.md). No secrets, no absolute paths. Fix a link there when a doc path moves. Do not rewrite recorded lessons to match a doc edit.

Graph READMEs under `video_buddy/workflows/` stay next to the JSON. Point them at `docs/` instead of copying policy.

Tests that pin a sentence (the H3 voice warning, Pack C local-only) must keep matching the file that owns that sentence. From `video_buddy/`:

```bash
python -m pytest tests/test_h3_catalog.py::test_h3_r2v_voice_reference_label_is_on_picker_surfaces tests/test_seedance_partner_pointers.py::test_local_only_policy_is_closed tests/test_model_paths.py::test_external_mode_docs_cover_the_contract tests/test_hermes_skill.py -q
```

Do not commit weights, `.env`, `state/runs`, or Comfy portable trees.
