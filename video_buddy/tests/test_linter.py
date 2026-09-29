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


def test_dynamic_combo_requires_prefixed_children():
    """COMFY_DYNAMICCOMBO_V3 children live under options, as parent.child."""
    from master_agent.comfy.validator import validate_workflow

    resize = [
        "COMFY_DYNAMICCOMBO_V3",
        {
            "options": [
                {
                    "key": "scale dimensions",
                    "inputs": {
                        "required": {
                            "width": ["INT", {"min": 0, "max": 4096}],
                            "height": ["INT", {"min": 0, "max": 4096}],
                            "crop": ["COMBO", {"options": ["disabled", "center"]}],
                        }
                    },
                },
                {
                    "key": "match size",
                    "inputs": {
                        "required": {
                            "match": ["IMAGE,MASK", {}],
                            "crop": ["COMBO", {"options": ["disabled", "center"]}],
                        }
                    },
                },
            ]
        },
    ]
    info = {
        "Src": {"input": {"required": {}}, "output": ["IMAGE"]},
        "ResizeImageMaskNode": {
            "input": {
                "required": {
                    "input": ["IMAGE", {}],
                    "resize_type": resize,
                    "scale_method": ["COMBO", {"options": ["area", "lanczos"]}],
                }
            },
            "output": ["IMAGE"],
        },
    }
    plain = {
        "1": {"class_type": "Src", "inputs": {}},
        "2": {
            "class_type": "ResizeImageMaskNode",
            "inputs": {
                "input": ["1", 0],
                "resize_type": "scale dimensions",
                "scale_method": "area",
                "width": 768,
                "height": 448,
            },
        },
    }
    report = validate_workflow(plain, info, file_label="plain-size")
    missing = {err.input_name for err in report.errors}
    assert missing == {"resize_type.width", "resize_type.height", "resize_type.crop"}

    prefixed = {
        "1": {"class_type": "Src", "inputs": {}},
        "2": {
            "class_type": "ResizeImageMaskNode",
            "inputs": {
                "input": ["1", 0],
                "resize_type": "scale dimensions",
                "scale_method": "lanczos",
                "resize_type.width": 384,
                "resize_type.height": 224,
                "resize_type.crop": "disabled",
            },
        },
    }
    ok = validate_workflow(prefixed, info, file_label="prefixed")
    assert ok.ok, ok.errors

    matched = {
        "1": {"class_type": "Src", "inputs": {}},
        "2": {
            "class_type": "ResizeImageMaskNode",
            "inputs": {
                "input": ["1", 0],
                "resize_type": "match size",
                "scale_method": "area",
                "resize_type.match": ["1", 0],
                "resize_type.crop": "center",
            },
        },
    }
    match_report = validate_workflow(matched, info, file_label="match")
    assert match_report.ok, match_report.errors


def _savevideo_0351_schema() -> dict:
    """ComfyUI 0.35.1 SaveVideo: format is a dynamic combo requiring codec.

    A second top-level codec widget stays optional so older flat prompts
    (``format`` + ``codec``) still queue.
    """
    codec = [
        "COMFY_DYNAMICCOMBO_V3",
        {
            "options": [
                {"key": "auto", "inputs": {}},
                {"key": "h264", "inputs": {}},
                {"key": "av1", "inputs": {}},
            ]
        },
    ]
    return {
        "input": {
            "required": {
                "video": ["VIDEO", {}],
                "filename_prefix": ["STRING", {"default": "video/ComfyUI"}],
                "format": [
                    "COMFY_DYNAMICCOMBO_V3",
                    {
                        "options": [
                            {
                                "key": key,
                                "inputs": {"required": {"codec": codec}},
                            }
                            for key in ("auto", "mp4", "mkv", "webm")
                        ]
                    },
                ],
            },
            "optional": {"codec": codec},
            "hidden": {
                "prompt": ["PROMPT", {}],
                "extra_pnginfo": ["EXTRA_PNGINFO", {}],
            },
        },
        "output": ["VIDEO"],
    }


def test_savevideo_accepts_flat_or_dotted_codec():
    """format.codec or the legacy flat codec both satisfy SaveVideo."""
    from master_agent.comfy.validator import validate_workflow

    info = {
        "Src": {"input": {"required": {}}, "output": ["VIDEO"]},
        "SaveVideo": _savevideo_0351_schema(),
    }

    def _graph(**codec_inputs):
        inputs = {
            "video": ["1", 0],
            "filename_prefix": "ltx23_inoutpaint",
            "format": "auto",
        }
        inputs.update(codec_inputs)
        return {
            "1": {"class_type": "Src", "inputs": {}},
            "78": {"class_type": "SaveVideo", "inputs": inputs},
        }

    missing = validate_workflow(_graph(), info, file_label="missing-codec")
    assert any(err.input_name == "format.codec" for err in missing.errors)

    flat = validate_workflow(_graph(codec="auto"), info, file_label="flat-codec")
    assert flat.ok, flat.errors

    dotted = validate_workflow(
        _graph(**{"format.codec": "auto"}), info, file_label="dotted-codec"
    )
    assert dotted.ok, dotted.errors

    both = validate_workflow(
        _graph(**{"format.codec": "auto", "codec": "auto"}),
        info,
        file_label="both-codecs",
    )
    assert both.ok, both.errors


def test_all_shipped_workflows_pass_hard_gate():
    """Every shipped workflow clears the SaveVideo hard gate.

    The cache still describes SaveVideo.format as a flat combo. ComfyUI
    0.35.1 makes format a dynamic combo that requires format.codec and
    still accepts the legacy flat codec. Overlaying that schema must not
    add a hard error on any shipped graph, and every graph the cache
    already accepts must still pass hard_gate.
    """
    import copy
    import json

    from master_agent.comfy.validator import ValidationReport, validate_workflow
    from master_agent.config import OBJECT_INFO_CACHE, WORKFLOW_FILES, WORKFLOWS_DIR

    cache = json.loads(OBJECT_INFO_CACHE.read_text(encoding="utf-8"))
    live = copy.deepcopy(cache)
    live["SaveVideo"] = _savevideo_0351_schema()

    clean: list[str] = []
    preexisting: list[str] = []
    for slug, rel in sorted(WORKFLOW_FILES.items()):
        workflow = json.loads((WORKFLOWS_DIR / rel).read_text(encoding="utf-8"))
        base = validate_workflow(
            copy.deepcopy(workflow), cache, file_label=rel, object_info_source="cache"
        )
        report = validate_workflow(
            copy.deepcopy(workflow), live, file_label=rel, object_info_source="savevideo-0.35.1"
        )
        save_gate = ValidationReport(file=rel, object_info_source="savevideo-0.35.1")
        save_gate.errors = [
            err for err in report.errors if err.input_name.startswith("format.codec")
        ]
        hard_gate(save_gate)
        added = {(e.node_id, e.input_name, e.message) for e in report.errors} - {
            (e.node_id, e.input_name, e.message) for e in base.errors
        }
        assert not added, f"{slug} gained hard errors under SaveVideo 0.35.1: {added}"
        if base.ok:
            hard_gate(report)
            clean.append(slug)
        else:
            preexisting.append(slug)

    assert len(clean) + len(preexisting) == len(WORKFLOW_FILES)
    assert "ltx23_inoutpaint" in clean
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
