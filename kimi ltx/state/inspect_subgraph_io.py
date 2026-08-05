"""Inspect how subgraph instance nodes in the CCC 4.1 UI workflow map to the
converted API file: for a given instance id, list its external input sources
and whether that source node (or its flattened descendants) exists in the API.

Usage: python state/inspect_subgraph_io.py <ui.json> <api.json> <instance_id> [<internal_node_id>...]
"""
import json
import sys


def main():
    ui_path, api_path, inst = sys.argv[1], sys.argv[2], sys.argv[3]
    internals = [int(x) for x in sys.argv[4:]]
    ui = json.load(open(ui_path, encoding="utf-8"))
    api = json.load(open(api_path, encoding="utf-8"))
    links = {l[0]: l for l in ui["links"]}
    for n in ui["nodes"]:
        if str(n["id"]) == inst:
            print(f"instance {inst} subgraph {n['type'][:8]}")
            for i in n.get("inputs", []):
                l = links.get(i.get("link"))
                if not l:
                    print(f"  slot {i.get('name')}: unlinked")
                    continue
                in_api = str(l[1]) in api or any(
                    k.startswith(f"{l[1]}:") for k in api
                )
                print(f"  slot {i.get('name')}: <- node {l[1]} out {l[2]} (in api: {in_api})")
    if internals:
        sub = None
        for s in ui["definitions"]["subgraphs"]:
            if any(x["id"] == inst for x in []):  # placeholder, match by nodes below
                pass
            ids = {x["id"] for x in s["nodes"]}
            if set(internals) & ids:
                sub = s
                break
        if sub:
            slinks = {l["id"]: l for l in sub["links"]}
            for n in sub["nodes"]:
                if n["id"] in internals:
                    print(f"== internal {n['id']} {n['type']} widgets: {n.get('widgets_values')}")
                    for i in n.get("inputs", []):
                        l = slinks.get(i.get("link"))
                        if l:
                            print(f"   {i['name']} <- {l['origin_id']} slot {l['origin_slot']}")


if __name__ == "__main__":
    main()
