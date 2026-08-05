"""Print UI widgets_values for given node ids across workflow files, plus the
live /object_info schema for their class, to build repair mappings.

Usage: python state/probe_widgets.py <ui.json> <id> [<id>...]
"""
import json
import sys
import urllib.request


def schema(cls):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:8188/object_info/{cls}") as r:
            d = json.load(r)[cls]["input"]
        req = list(d.get("required", {}))
        opt = list(d.get("optional", {}))
        print(f"   schema req={req} opt={opt}")
    except Exception as e:
        print(f"   schema error: {e}")


def main():
    ui_path = sys.argv[1]
    ids = {int(x) for x in sys.argv[2:]}
    ui = json.load(open(ui_path, encoding="utf-8"))
    stacks = [ui.get("nodes", [])]
    stacks += [s.get("nodes", []) for s in ui.get("definitions", {}).get("subgraphs", [])]
    for nodes in stacks:
        for n in nodes:
            if n.get("id") in ids:
                print(f"{n['id']} {n['type']} widgets={n.get('widgets_values')}")
                schema(n["type"])


if __name__ == "__main__":
    main()
