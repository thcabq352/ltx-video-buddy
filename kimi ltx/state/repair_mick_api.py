"""One-shot repair of converter widget-misalignment in the Mickmumpitz *_api.json files.

The workflow-to-api converter maps widgets_values positionally; where a pack's
current INPUT_TYPES drifted from the workflow's version (or linked widgets were
dropped), values landed on the wrong keys. This rewrites the affected nodes from
the ORIGINAL UI files' widgets_values, mapped onto the current node schemas
(verified against state/object_info.json + pack sources).

Idempotent: safe to re-run.
"""
import json

REPAIRS = {
    "workflows/260713_MICKMUMPITZ_DATASET-TAGGER_1-5_api.json": {
        "104": {  # CCC_ShowImageTextPairs — old 5-widget version vs current 4
            "draw_boxes": True,
            "bbox_format": "normalized_1000_xyxy",
            "training_resolution": "off",
            "bucket_divisibility": 64,
        },
        "70:103": {  # Save Text File (inside subgraph instance 70) — shifted by dropped link widgets
            "filename_delimiter": "",
            "filename_number_padding": 0,
            "file_extension": ".json",
            "encoding": "utf-8",
            "filename_suffix": "",
        },
    },
    "workflows/260713_MICKMUMPITZ_TAG-REVIEW-EDIT_1-0_api.json": {
        "16": {  # CCC_DatasetReviewer
            "bbox_format": "yxyx_normalized",
            "edited_data_json": "",
        },
        "9": {  # Save Text File
            "filename_delimiter": "",
            "filename_number_padding": 0,
            "file_extension": ".json",
            "encoding": "utf-8",
            "filename_suffix": "",
        },
    },
    "workflows/260720_MICKMUMPITZ_CCC_4-01_ADV_api.json": {
        "1560": {  # ImageCollectUnpack — old order (source, fit, target_size); 'Dataset' choice removed
            "source": "first",
            "target_size": "first",
            "fit": "stretch",
        },
    },
}

# JoinStringMulti nodes in CCC ADV: widgets [3, " ", false, null] got shifted.
CCC_API = "workflows/260720_MICKMUMPITZ_CCC_4-01_ADV_api.json"


def main():
    for path, fixes in REPAIRS.items():
        with open(path, encoding="utf-8") as f:
            wf = json.load(f)
        changed = 0
        for nid, vals in fixes.items():
            if nid in wf:
                wf[nid]["inputs"].update(vals)
                changed += 1
            else:
                print(f"WARN {path}: node {nid} not found")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(wf, f, ensure_ascii=False, indent=2)
        print(f"{path}: {changed} node(s) repaired")

    with open(CCC_API, encoding="utf-8") as f:
        wf = json.load(f)
    n = 0
    for nid, node in wf.items():
        if node.get("class_type") == "JoinStringMulti":
            inp = node["inputs"]
            if not isinstance(inp.get("delimiter"), str) or inp.get("return_list") is None:
                inp["delimiter"] = " "
                inp["return_list"] = False
                n += 1
    with open(CCC_API, "w", encoding="utf-8") as f:
        json.dump(wf, f, ensure_ascii=False, indent=2)
    print(f"{CCC_API}: {n} JoinStringMulti node(s) repaired")


if __name__ == "__main__":
    main()
