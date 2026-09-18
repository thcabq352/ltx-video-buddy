# Brain / Hands (`buddy.capability.contract/v1`)

Director (brain) ranks **stories**. Hands answers whether the live tower
can fulfill that story **right now**. Capacity math stays out of routing.

## Contract

Source of truth for this tree is this file plus
`master_agent/capability.py` / `master_agent/hands.py`:

- `CapabilityContract` / `Hands.can_fulfill` → `FitResult`
- 8s last-frame chaining owned by Hands (`plan_last_frame_chain`)

Field names match `buddy.capability.contract/v1`. If a sibling Rust
struct lands extra keys, add them without dropping these. Never serialize
VRAM, slot, or weight-path fields on the contract.

## Python live orchestrator

| Piece | Where |
|---|---|
| `CapabilityContract` | `master_agent/capability.py` — `buddy.capability.contract/v1` |
| `Hands.can_fulfill` | `master_agent/hands.py` — `SnapshotHands` / `LiveHands` → `FitResult` |
| Story rank then fit | `orchestrator/director.py` — `rank_story_candidates` → Hands |
| 8s last-frame chain | `hands.plan_last_frame_chain` (20s → 3×8s); pipeline uses it |

Brain writes a contract for what one run guarantees:

```json
{
  "schema": "buddy.capability.contract/v1",
  "model": "ltx-2.5",
  "family": "ltx25",
  "variant": "ltx25_t2v_i2v",
  "resolution": [768, 512],
  "duration_s": 8.0,
  "story_duration_s": 20.0,
  "audio": true,
  "control_layers": ["openpose"]
}
```

Hands reads a live / cached `/object_info` + inventory + VRAM snapshot and
returns a FitResult (`ok` / `insufficient_vram` / `missing_weights` /
`missing_nodes`). Inject `SnapshotHands(HandsSnapshot(...))` in tests.

Long `story_duration_s` is **not** a brain VRAM split. Hands plans an 8s
last-frame chain and (when a prior clip exists) extracts the last frame
for the next I2V burn. Music-video mode keeps beat windows and does
**not** use this chain.

## Do not

- Pick “cheaper GPU” graphs inside the director / LLM payload
- Put `vram*` / `slot` / `weight_path` on the contract
- Remap Kling, download weights, or auto-install Comfy packs
- Break TeaCache inject-when-registered, provenance sidecars, or MTV mode

Tests (no GPU): `python -m pytest tests/test_brain_hands.py -q`
