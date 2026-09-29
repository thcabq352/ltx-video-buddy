"""Workflow linter hard gate. Run: .venv/Scripts/python.exe -m pytest tests/test_linter.py -q"""

import pytest

from master_agent.comfy.linter import LintBlocked, hard_gate, lint_workflow
from master_agent.comfy.validator import ValidationReport


def test_linter_blocks_unknown_class():
    wf = {"1": {"class_type": "NotANode", "inputs": {}}}
    report = lint_workflow(wf, object_info={}, file_label="bad.json")
    assert report.ok is False
    with pytest.raises(LintBlocked):
        hard_gate(report)
    clean = ValidationReport(file="ok")
    assert hard_gate(clean) is clean


def test_v3_combo_and_lipsync_placeholder():
    from master_agent.comfy.validator import validate_workflow
    from master_agent.comfy.workflow_patcher import load_workflow_template

    info = {
        "LoadVideo": {
            "input": {
                "required": {
                    "file": ["COMBO", {"options": ["a.mp4", "warehouse_src_30fps.mp4"]}],
                }
            },
            "output": ["VIDEO"],
        }
    }
    bad = {
        "1": {
            "class_type": "LoadVideo",
            "inputs": {"file": "other.mp4"},
        }
    }
    report = validate_workflow(bad, info, file_label="combo")
    assert any("not in combo choices" in err.message for err in report.errors)

    lipsync = load_workflow_template("lipsync")
    held = validate_workflow(lipsync, info, file_label="lipsync")
    assert any("warehouse_src_30fps.mp4" in err.message for err in held.errors)


def test_autogrow_slots_and_combo_choices_alias():
    """Dotted autogrow slots use the template type; V3 COMBO accepts `choices`."""
    import copy

    from master_agent.comfy.validator import validate_workflow

    autogrow = [
        "COMFY_AUTOGROW_V3",
        {
            "template": {
                "input": {"optional": {"image": ["IMAGE", {}]}},
                "prefix": "image",
                "min": 0,
                "max": 4,
            }
        },
    ]
    info = {
        "LoadImage": {
            "input": {"required": {"image": ["STRING", {}]}},
            "output": ["IMAGE"],
        },
        "Sink": {
            "input": {
                "required": {
                    "imgs": autogrow,
                    "mode": ["COMBO", {"choices": ["a", "b"]}],
                }
            },
            "output": ["*"],
        },
    }
    ok = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
        "2": {
            "class_type": "Sink",
            "inputs": {"imgs.image0": ["1", 0], "mode": "a"},
        },
    }
    report = validate_workflow(ok, info, file_label="autogrow")
    assert report.ok, report.errors

    audio = copy.deepcopy(info)
    audio["LoadImage"]["output"] = ["AUDIO"]
    mismatched = validate_workflow(ok, audio, file_label="autogrow-type")
    assert any("type mismatch" in err.message and "IMAGE" in err.message for err in mismatched.errors)

    bare = copy.deepcopy(ok)
    bare["2"]["inputs"]["imgs"] = ["1", 0]
    del bare["2"]["inputs"]["imgs.image0"]
    bare_report = validate_workflow(bare, info, file_label="autogrow-bare")
    assert any("COMFY_AUTOGROW_V3" in err.message for err in bare_report.errors)

    bad_combo = copy.deepcopy(ok)
    bad_combo["2"]["inputs"]["mode"] = "nope"
    combo_report = validate_workflow(bad_combo, info, file_label="combo-alias")
    assert any("not in combo choices" in err.message for err in combo_report.errors)


def _savevideo_035_schema() -> dict:
    """ComfyUI 0.35.1 SaveVideo: format.codec is required, flat codec still queues."""
    codec = [
        "COMFY_DYNAMICCOMBO_V3",
        {
            "options": [
                {"key": "auto", "inputs": {"required": {}}},
                {
                    "key": "h264",
                    "inputs": {
                        "optional": {
                            "encoding": ["COMBO", {"options": ["auto", "re-encode"]}],
                        }
                    },
                },
                {
                    "key": "av1",
                    "inputs": {
                        "optional": {
                            "encoding": ["COMBO", {"options": ["auto", "re-encode"]}],
                        }
                    },
                },
            ]
        },
    ]
    return {
        "SaveVideo": {
            "input": {
                "required": {
                    "video": ["VIDEO", {}],
                    "filename_prefix": ["STRING", {}],
                    "format": [
                        "COMFY_DYNAMICCOMBO_V3",
                        {
                            "options": [
                                {"key": "auto", "inputs": {"required": {"codec": codec}}},
                                {"key": "mp4", "inputs": {"required": {"codec": codec}}},
                                {"key": "webm", "inputs": {"required": {"codec": codec}}},
                            ]
                        },
                    ],
                },
                "hidden": {"codec": codec},
            },
            "output": ["VIDEO"],
        }
    }


def test_savevideo_accepts_flat_codec_or_format_codec():
    from master_agent.comfy.validator import validate_workflow

    info = _savevideo_035_schema()
    flat = {
        "79": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["1", 0],
                "filename_prefix": "clip",
                "format": "auto",
                "codec": "auto",
            },
        }
    }
    # video link target is absent; only the format.codec gate is under test
    flat_report = validate_workflow(flat, info, file_label="flat-codec")
    assert not any(err.input_name == "format.codec" for err in flat_report.errors), flat_report.errors
    dotted = {
        "79": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["1", 0],
                "filename_prefix": "clip",
                "format": "auto",
                "format.codec": "auto",
            },
        }
    }
    dotted_report = validate_workflow(dotted, info, file_label="dotted-codec")
    assert not any(err.input_name == "format.codec" for err in dotted_report.errors), dotted_report.errors
    neither = {
        "79": {
            "class_type": "SaveVideo",
            "inputs": {"video": ["1", 0], "filename_prefix": "clip", "format": "auto"},
        }
    }
    missing = validate_workflow(neither, info, file_label="no-codec")
    assert any(err.input_name == "format.codec" for err in missing.errors)
    resize = {
        "34": {
            "class_type": "ResizeImageMaskNode",
            "inputs": {
                "resize_type": "scale dimensions",
                "width": 384,
                "height": 224,
            },
        }
    }
    resize_info = {
        "ResizeImageMaskNode": {
            "input": {
                "required": {
                    "resize_type": [
                        "COMFY_DYNAMICCOMBO_V3",
                        {
                            "options": [
                                {
                                    "key": "scale dimensions",
                                    "inputs": {
                                        "required": {
                                            "width": ["INT", {"min": 0, "max": 8192}],
                                            "height": ["INT", {"min": 0, "max": 8192}],
                                            "crop": ["COMBO", {"options": ["disabled", "center"]}],
                                        }
                                    },
                                }
                            ]
                        },
                    ]
                }
            },
            "output": ["IMAGE"],
        }
    }
    resize_report = validate_workflow(resize, resize_info, file_label="flat-width")
    assert any(err.input_name == "resize_type.width" for err in resize_report.errors)


def test_every_shipped_workflow_passes_savevideo_hard_gate():
    """A 0.35.1 SaveVideo schema must not add hard errors on shipped graphs."""
    import json

    from master_agent.comfy.linter import hard_gate
    from master_agent.comfy.validator import _unwrap_workflow, validate_workflow
    from master_agent.config import OBJECT_INFO_CACHE, WORKFLOW_FILES, WORKFLOWS_DIR

    cached = json.loads(OBJECT_INFO_CACHE.read_text(encoding="utf-8"))
    live = json.loads(json.dumps(cached))
    save = live["SaveVideo"]
    schema = _savevideo_035_schema()["SaveVideo"]["input"]
    # Keep every other SaveVideo input the cache already knows.
    save_input = save.setdefault("input", {})
    save_input.setdefault("required", {})["format"] = schema["required"]["format"]
    codec_spec = schema["hidden"]["codec"]
    for section in ("required", "optional"):
        block = save_input.get(section)
        if isinstance(block, dict):
            block.pop("codec", None)
    save_input.setdefault("hidden", {})["codec"] = codec_spec

    checked = 0
    for slug, rel in sorted(WORKFLOW_FILES.items()):
        path = WORKFLOWS_DIR / rel
        raw = json.loads(path.read_text(encoding="utf-8"))
        workflow = _unwrap_workflow(raw, rel)
        before = validate_workflow(workflow, cached, file_label=slug)
        after = validate_workflow(workflow, live, file_label=slug)
        before_keys = {(err.node_id, err.input_name, err.message) for err in before.errors}
        added = [
            err
            for err in after.errors
            if (err.node_id, err.input_name, err.message) not in before_keys
        ]
        assert not added, f"{slug} gained hard errors: {added}"
        assert not any(err.input_name == "format.codec" for err in after.errors), slug
        if before.ok:
            hard_gate(after)
        checked += 1
    assert checked == len(WORKFLOW_FILES)
    # The ltx25 graph itself is inside that set and carries the dotted key.
    ltx = _unwrap_workflow(
        json.loads((WORKFLOWS_DIR / WORKFLOW_FILES["ltx25_inoutpaint"]).read_text(encoding="utf-8")),
        "ltx25_inoutpaint",
    )
    assert ltx["79"]["inputs"]["format.codec"] == "auto"
    report = validate_workflow(ltx, live, file_label="ltx25_inoutpaint")
    save_errors = [err for err in report.errors if err.node_id == "79"]
    assert not save_errors, save_errors
    hard_gate_errors = [err for err in report.errors if err.input_name.startswith("format.")]
    assert not hard_gate_errors
