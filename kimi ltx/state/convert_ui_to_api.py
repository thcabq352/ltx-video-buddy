"""Convert a ComfyUI UI-format workflow (nodes/links/definitions.subgraphs)
into API/prompt format (dict keyed by node id, {class_type, inputs}).

Conventions reverse-engineered from the existing UI->API pairs in workflows/
(see e.g. 260608_VIDEO-BUDDY_AI-VFX_1-3_ADV.json vs its _api.json):

  - Frontend-only classes are dropped: Reroute, SetNode, GetNode,
    MarkdownNote, Note, Mickmumpitz(Multiline)Label, Label (rgthree),
    Node Bypasser, GroupNode.
  - Bypassed (mode 4) / muted (mode 2) nodes are dropped; consumers are
    resolved THROUGH them when exactly one linked input matches the consumed
    output type (bypass = pass-through), otherwise the input is dropped.
  - Nodes with no outgoing links are dropped unless the class is an
    output_node in /object_info.
  - GetNode/SetNode pairs are unwrapped: a GetNode reference resolves to the
    upstream source of the SetNode with the same constant name.
  - Reroutes are resolved transitively.
  - Subgraph instances (node type is a UUID in definitions.subgraphs) are
    expanded inline: inner nodes get ids "<instance_id>:<inner_id>".
    Instance input slot k (matched by subgraph input name) feeds inner links
    originating at the virtual input node (-10) slot k; instance output
    slot k maps to the inner link feeding the virtual output node (-20)
    slot k. An unconnected output slot yields a dangling [instance_id, slot]
    reference (faithful to the original converter; repair passes clean up).
  - Widgets: widgets_values map onto the class schema's widget inputs
    (primitive types + combos, required then optional, declaration order).
    Dict-form widgets_values map by name; list-form map positionally,
    consuming a slot for widget inputs converted to links and skipping the
    hidden control_after_generate value after seed widgets. Widget values
    come first in the emitted inputs dict, then link inputs (both in
    schema declaration order).
  - Unlinked non-widget inputs are omitted; _meta.title is the UI node
    title or the class display_name.

Known converter quirks (widget drift on schema-version mismatch, dangling
subgraph-output links) are deliberately NOT repaired here — see
state/repair_batch1_api.py / state/repair_batch2_api.py.

Usage:
    python state/convert_ui_to_api.py <ui.json> [<api.json>]
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBJECT_INFO_CACHE = os.path.join(ROOT, "state", "object_info.json")
COMFY_OBJECT_INFO = "http://127.0.0.1:8188/object_info"

DROP_TYPES = {
    "Reroute",
    "SetNode",
    "GetNode",
    "MarkdownNote",
    "Note",
    "MickmumpitzLabel",
    "MickmumpitzMultilineLabel",
    "Label (rgthree)",
    "Node Bypasser",
    "GroupNode",
}
PRIMITIVES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
SUBGRAPH_INPUT_ID = -10
SUBGRAPH_OUTPUT_ID = -20

warnings: list[str] = []


def warn(msg: str) -> None:
    warnings.append(msg)
    print(f"WARN: {msg}", file=sys.stderr)


def load_object_info() -> dict:
    try:
        with urllib.request.urlopen(COMFY_OBJECT_INFO, timeout=3) as r:
            return json.load(r)
    except Exception:
        with open(OBJECT_INFO_CACHE, encoding="utf-8") as f:
            return json.load(f)


def norm_links(links: list) -> dict:
    """link_id -> (src_id, src_slot, dst_id, dst_slot); accepts array or
    dict link records (top-level uses arrays, subgraphs use dicts)."""
    out = {}
    for l in links or []:
        if isinstance(l, dict):
            out[l["id"]] = (
                l["origin_id"],
                l["origin_slot"],
                l["target_id"],
                l["target_slot"],
            )
        else:
            out[l[0]] = (l[1], l[2], l[3], l[4])
    return out


def is_widget_spec(spec) -> bool:
    if not isinstance(spec, (list, tuple)) or not spec:
        return False
    if spec_options(spec).get("forceInput"):
        return False  # input-only in the frontend; never occupies a widget slot
    return (
        isinstance(spec[0], (list, tuple))
        or spec[0] in PRIMITIVES
        or spec[0] == "COMFY_DYNAMICCOMBO_V3"
    )


def dynamic_combo_children(spec, key) -> list:
    """Widget-typed child input names of the selected dynamic-combo option."""
    for opt in spec_options(spec).get("options") or []:
        if opt.get("key") == key:
            out = []
            inputs = opt.get("inputs") or {}
            for section in ("required", "optional"):
                for cname, cspec in (inputs.get(section) or {}).items():
                    if is_widget_spec(cspec):
                        out.append(cname)
            return out
    return []


def spec_options(spec) -> dict:
    if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1]
    return {}


def coerce_widget_value(spec, value):
    """Fit a widget value to its declared primitive (the frontend stores some
    string widgets as numbers, e.g. OlmDragCrop's drawing_version nonce)."""
    if (
        isinstance(spec, (list, tuple))
        and spec
        and spec[0] == "STRING"
        and isinstance(value, (int, float, bool))
    ):
        return str(value)
    return value


def const_name(node) -> object:
    """Constant name of a SetNode/GetNode from its widgets_values."""
    wv = node.get("widgets_values")
    if isinstance(wv, dict):
        return wv.get("constant", wv.get("CONSTANT"))
    if isinstance(wv, list) and wv:
        return wv[0]
    return None


class Expansion:
    """One graph level being converted: top level or one subgraph instance."""

    def __init__(self, conv, subdef, inst_node, prefix):
        self.conv = conv
        self.subdef = subdef  # None at top level
        self.inst_node = inst_node  # None at top level
        self.prefix = prefix
        nodes = subdef["nodes"] if subdef else conv.ui.get("nodes", [])
        links = subdef.get("links", []) if subdef else conv.ui.get("links", [])
        self.nodes = {n["id"]: n for n in nodes}
        self.links = norm_links(links)

    def resolve_instance_input(self, slot: int):
        """Resolve subgraph input slot to this instance's external source."""
        inputs = self.subdef.get("inputs", [])
        if slot >= len(inputs):
            return None
        name = inputs[slot].get("name")
        ui_inputs = self.inst_node.get("inputs", [])
        inp = next((i for i in ui_inputs if i.get("name") == name), None)
        link = inp.get("link") if inp else None
        if link is None:
            return None
        rec = self.conv.top.links.get(link)
        if not rec:
            return None
        return self.conv.resolve(self.conv.top, rec[0], rec[1])


class Converter:
    def __init__(self, ui: dict, object_info: dict):
        self.ui = ui
        self.oi = object_info
        self.subdefs = {
            s["id"]: s for s in ui.get("definitions", {}).get("subgraphs", [])
        }
        self.top = Expansion(self, None, None, "")
        # SetNode constant -> list of (exp, node); GetNodes resolve through it
        self.setnodes: dict[object, list[tuple[Expansion, dict]]] = {}
        for n in self.top.nodes.values():
            if n.get("type") == "SetNode":
                self.setnodes.setdefault(const_name(n), []).append((self.top, n))
        # (SetNodes inside subgraph defs are registered lazily per expansion.)
        self.out: dict[str, dict] = {}

    # -- link resolution ----------------------------------------------------

    def resolve(self, exp: Expansion, node_id: int, slot: int, seen=()):
        """Resolve a link source to (api_node_id, out_slot) or None,
        chasing reroutes, get/set nodes, bypassed pass-throughs and
        subgraph-instance outputs."""
        if node_id in (SUBGRAPH_INPUT_ID, SUBGRAPH_OUTPUT_ID):
            return None
        node = exp.nodes.get(node_id)
        if node is None:
            return None
        if (node_id, slot) in seen:
            return None
        seen = seen + ((node_id, slot),)
        ntype = node.get("type")

        if ntype == "Reroute" or ntype == "SetNode":
            return self._through(exp, node, seen)
        if ntype == "GetNode":
            entries = self.setnodes.get(const_name(node))
            if not entries:
                warn(f"GetNode {node_id}: no SetNode for constant "
                     f"{const_name(node)!r}")
                return None
            if len(entries) > 1:
                warn(f"GetNode {node_id}: constant {const_name(node)!r} has "
                     f"{len(entries)} SetNodes; using the first")
            sexp, snode = entries[0]
            return self._through(sexp, snode, seen)

        if node.get("mode", 0) in (2, 4):
            # bypassed/muted real node: pass through the single linked input
            # whose type matches the consumed output slot
            out_type = None
            outputs = node.get("outputs") or []
            if slot < len(outputs):
                out_type = outputs[slot].get("type")
            cands = []
            for i in node.get("inputs", []):
                if i.get("link") is None:
                    continue
                if out_type is None or i.get("type") in (out_type, "*"):
                    cands.append(i["link"])
            if len(cands) == 1:
                rec = exp.links.get(cands[0])
                if rec:
                    return self.resolve(exp, rec[0], rec[1], seen)
            return None

        if ntype in self.subdefs:
            # link out of a subgraph instance: map output slot through -20
            sub = self.subdefs[ntype]
            sub_exp = Expansion(self, sub, node, f"{node_id}:")
            for lid, rec in sub_exp.links.items():
                if rec[2] == SUBGRAPH_OUTPUT_ID and rec[3] == slot:
                    if rec[0] == SUBGRAPH_INPUT_ID:
                        return sub_exp.resolve_instance_input(rec[1])
                    return self.resolve(sub_exp, rec[0], rec[1], seen)
            # output slot unconnected inside: faithful dangling reference
            warn(f"instance {node_id}: subgraph output slot {slot} "
                 "unconnected inside; emitting dangling reference")
            return (str(node_id), slot)

        return (f"{exp.prefix}{node_id}", slot)

    def _through(self, exp: Expansion, node: dict, seen):
        for i in node.get("inputs", []):
            if i.get("link") is None:
                continue
            rec = exp.links.get(i["link"])
            if rec:
                return self.resolve(exp, rec[0], rec[1], seen)
        return None

    # -- widget mapping ------------------------------------------------------

    def widget_inputs(self, class_type: str):
        """[(name, spec)] widget inputs in declaration order (required+optional)."""
        info = self.oi.get(class_type) or {}
        spec_in = info.get("input") or {}
        out = []
        for section in ("required", "optional"):
            for name, spec in (spec_in.get(section) or {}).items():
                if is_widget_spec(spec):
                    out.append((name, spec))
        return out

    def map_widgets(self, node: dict, class_type: str, linked: set,
                    overrides: dict) -> dict:
        wv = node.get("widgets_values")
        widgets = self.widget_inputs(class_type)
        values = {}
        if isinstance(wv, dict):
            for name, spec in widgets:
                if name in overrides:
                    val = overrides[name]
                elif name in wv:
                    val = wv[name]
                elif "default" in spec_options(spec):
                    val = spec_options(spec)["default"]
                else:
                    continue
                if (
                    isinstance(spec, (list, tuple)) and spec
                    and spec[0] == "COMFY_DYNAMICCOMBO_V3"
                ):
                    if name not in linked:
                        values[name] = val
                    # dynamic-combo children ('resize_type.width', ...) map
                    # by dotted name right after the parent
                    for child in dynamic_combo_children(spec, val):
                        cname = f"{name}.{child}"
                        if cname in linked:
                            continue
                        if cname in wv:
                            values[cname] = wv[cname]
                    continue
                if name in linked:
                    continue
                values[name] = coerce_widget_value(spec, val)
        elif isinstance(wv, list):
            idx = 0
            for name, spec in widgets:
                if idx >= len(wv):
                    if name not in linked and "default" in spec_options(spec):
                        values[name] = spec_options(spec)["default"]
                    continue
                val = wv[idx]
                idx += 1
                # skip the hidden control_after_generate combo after seed
                # widgets: flagged in the spec, or unflagged (Impact pack)
                # but present as 'fixed'/'increment'/... in the values
                cag = spec_options(spec).get("control_after_generate") or (
                    isinstance(name, str) and name.endswith("seed")
                )
                if cag and idx < len(wv) and wv[idx] in (
                    "fixed", "increment", "decrement", "randomize",
                ):
                    idx += 1
                if (
                    isinstance(spec, (list, tuple)) and spec
                    and spec[0] == "COMFY_DYNAMICCOMBO_V3"
                ):
                    if name not in linked:
                        values[name] = val
                    for child in dynamic_combo_children(spec, val):
                        cname = f"{name}.{child}"
                        if idx >= len(wv):
                            break
                        cval = wv[idx]
                        idx += 1
                        if cname in linked:
                            continue  # slot consumed, link wins
                        values[cname] = cval
                    continue
                if name in linked:
                    continue  # slot consumed, link wins
                if name in overrides:
                    val = overrides[name]
                values[name] = coerce_widget_value(spec, val)
        return values

    # -- node emission -------------------------------------------------------

    def keep_node(self, exp: Expansion, node: dict, outgoing: set) -> bool:
        if node.get("type") in DROP_TYPES:
            return False
        if node.get("mode", 0) in (2, 4):
            return False
        if node["id"] not in outgoing and not (
            (self.oi.get(node.get("type")) or {}).get("output_node")
        ):
            return False
        return True

    def emit_node(self, exp: Expansion, node: dict, overrides: dict) -> None:
        class_type = node.get("type")
        api_id = f"{exp.prefix}{node['id']}"
        schema_known = class_type in self.oi

        link_map: dict[str, list] = {}
        ui_order: list[str] = []
        widget_names = {n for n, _ in self.widget_inputs(class_type)}
        for i in node.get("inputs", []):
            name = i.get("name")
            ui_order.append(name)
            if i.get("link") is None:
                continue
            rec = exp.links.get(i["link"])
            if not rec:
                continue
            if rec[0] == SUBGRAPH_INPUT_ID:
                r = exp.resolve_instance_input(rec[1])
            else:
                r = self.resolve(exp, rec[0], rec[1])
            if r is None:
                if name not in widget_names:
                    # non-widget inputs get dropped; widget inputs silently
                    # fall back to their widget value below
                    warn(f"{api_id}.{name}: input link could not be "
                         "resolved; dropped")
                continue
            link_map[name] = [r[0], r[1]]

        widgets = self.map_widgets(node, class_type, set(link_map), overrides)

        inputs: dict[str, object] = {}
        if schema_known:
            info = self.oi[class_type]
            spec_in = info.get("input") or {}
            order = []
            for section in ("required", "optional"):
                order.extend((spec_in.get(section) or {}).keys())
            # widgets in mapping (declaration) order — includes dynamic-combo
            # children ('resize_type.width') that are not top-level schema keys
            for name, val in widgets.items():
                inputs[name] = val
            for name in order:
                if name in link_map:
                    inputs[name] = link_map[name]
            for name in ui_order:  # links unknown to the (drifted) schema
                if name in link_map and name not in inputs:
                    inputs[name] = link_map[name]
        else:
            if class_type not in DROP_TYPES:
                warn(f"{api_id}: class '{class_type}' not in object_info; "
                     "widgets left unmapped")
            for name in ui_order:
                if name in link_map:
                    inputs[name] = link_map[name]

        title = node.get("title")
        if not title:
            title = (self.oi.get(class_type) or {}).get("display_name") or class_type
        self.out[api_id] = {
            "inputs": inputs,
            "class_type": class_type,
            "_meta": {"title": title},
        }

    def emit_level(self, exp: Expansion) -> None:
        outgoing = {rec[0] for rec in exp.links.values()}
        nodes = exp.subdef["nodes"] if exp.subdef else self.ui.get("nodes", [])
        for node in nodes:
            ntype = node.get("type")
            if exp.subdef is None and ntype in self.subdefs:
                if node.get("mode", 0) in (2, 4):
                    warn(f"instance {node['id']} bypassed/muted; not expanded")
                    continue
                sub = self.subdefs[ntype]
                sub_exp = Expansion(self, sub, node, f"{node['id']}:")
                for n in sub_exp.nodes.values():
                    if n.get("type") == "SetNode":
                        self.setnodes.setdefault(const_name(n), []).append(
                            (sub_exp, n))
                overrides = self.proxy_overrides(node)
                inner_outgoing = {r[0] for r in sub_exp.links.values()}
                for inner in sub["nodes"]:
                    if self.keep_node(sub_exp, inner, inner_outgoing):
                        self.emit_node(
                            sub_exp, inner, overrides.get(inner["id"], {}))
                continue
            if self.keep_node(exp, node, outgoing):
                self.emit_node(exp, node, {})

    def proxy_overrides(self, inst_node: dict) -> dict:
        """Instance-level widgets_values applied onto proxied inner widgets,
        in properties.proxyWidgets order."""
        wv = inst_node.get("widgets_values")
        proxied = (inst_node.get("properties") or {}).get("proxyWidgets") or []
        out: dict[int, dict] = {}
        if not isinstance(wv, list) or not wv:
            return out
        for (inner_id, wname), val in zip(proxied, wv):
            if "." in str(wname):
                warn(f"instance {inst_node['id']}: proxied widget '{wname}' "
                     "unsupported; skipped")
                continue
            out.setdefault(int(inner_id), {})[wname] = val
        return out

    def run(self) -> dict:
        self.emit_level(self.top)
        return self.out


def convert(ui_path: str, api_path: str, object_info: dict | None = None) -> dict:
    with open(ui_path, encoding="utf-8") as f:
        ui = json.load(f)
    oi = object_info or load_object_info()
    api = Converter(ui, oi).run()
    with open(api_path, "w", encoding="utf-8") as f:
        json.dump(api, f, ensure_ascii=False, indent=2)
    return api


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    ui_path = sys.argv[1]
    api_path = sys.argv[2] if len(sys.argv) > 2 else ui_path.replace(
        ".json", "_api.json")
    api = convert(ui_path, api_path)
    print(f"{ui_path} -> {api_path}: {len(api)} nodes, "
          f"{len(warnings)} warning(s)")


if __name__ == "__main__":
    main()
