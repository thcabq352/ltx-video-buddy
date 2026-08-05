"""Dump external link resolutions for one flattened subgraph instance in an
API-format workflow: which outside node each internal consumer links to.

Usage: python state/inspect_api_instance.py <api.json> <instance_prefix> [node_id ...]
"""
import json
import sys


def main():
    api_path, inst = sys.argv[1], sys.argv[2]
    focus = sys.argv[3:]
    api = json.load(open(api_path, encoding="utf-8"))
    ext = {}
    for k, v in api.items():
        if not k.startswith(inst + ":"):
            continue
        for name, val in v.get("inputs", {}).items():
            if isinstance(val, list) and not str(val[0]).startswith(inst + ":"):
                ext.setdefault(str(val[0]), []).append(
                    (k.split(":", 1)[1], v["class_type"], name, val[1]))
    for src, uses in sorted(ext.items()):
        print(src, api.get(src, {}).get("class_type"), "->", uses[:5])
    for k in focus:
        if k in api:
            print(k, api[k]["class_type"], json.dumps(api[k]["inputs"])[:300])
        else:
            print(k, "NOT IN API")


if __name__ == "__main__":
    main()
