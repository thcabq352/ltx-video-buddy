"""One-shot repair of the batch-4 converted MOVIE-BUILDER api.json.

- private author assets referenced by LoadImage/LoadAudio -> repoint to
  existing placeholders in ComfyUI/input/ (front.png/face.png are the
  author's protagonist refs, brielle_ref.mp3 his voice sample)
- dangling subgraph-output links (unconnected instance output slots in the
  author's own UI file; the converter emits them faithfully) -> drop

Idempotent: safe to re-run.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "workflows/260507_MICKMUMPITZ_MOVIE-BUILDER_1-1_ADV_api.json"

REPOINT = {
    "1284": {"image": "ComfyUI_01828_.png"},
    "4867": {"image": "ComfyUI_01828_.png"},
    "10343": {"audio": "BusinessWoman_AUDIO.mp3"},
}


def drop_dead_links(wf):
    n = 0
    for node in wf.values():
        for key in list(node.get("inputs", {})):
            val = node["inputs"][key]
            if isinstance(val, list) and val and str(val[0]) not in wf:
                del node["inputs"][key]
                n += 1
    return n


def main():
    path = os.path.join(ROOT, API)
    with open(path, encoding="utf-8") as f:
        wf = json.load(f)
    changed = 0
    for nid, vals in REPOINT.items():
        if nid in wf:
            wf[nid]["inputs"].update(vals)
            changed += 1
        else:
            print(f"WARN {API}: node {nid} not found")
    dead = drop_dead_links(wf)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(wf, f, ensure_ascii=False, indent=2)
    print(f"{API}: {changed} node(s) repointed, {dead} dead link(s) dropped")


if __name__ == "__main__":
    main()
