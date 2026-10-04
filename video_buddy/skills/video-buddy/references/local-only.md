# Local-only hard requirement

Local-only is a hard requirement. Agents operating Pack C (Seedance Draft→Final) run video generation on this machine.

- Cloud APIs: forbidden
- Cloud services: forbidden
- Cloud-hosted inference: forbidden

| Piece | Value |
|---|---|
| Comfy | `http://127.0.0.1:8188` |
| Text / first frame | `ltx25_t2v_i2v` |
| First + last | `ltx25_flf2v` |
| References | `ltx25_msr` |

`seedance25_draft_{t2v,i2v,r2v}` and `api_seedance2_5_draft_{t2v,i2v,r2v}` record Partner field names (`Seedance 2.5 Draft`, `draft_task_id`, `ByteDance2DraftToFinalVideoNode`). Those class types call hosted inference. The agent does not load them, does not queue them, and does not call comfy.org, BytePlus, ModelArk, or KIE.

Wan and MiniMax H3 catalog ids stay local Comfy paths. Name one with `--variant` when that graph is the shot. Pack C does not reroute them through a cloud host.

Optional Grok may direct. It does not burn the video.
