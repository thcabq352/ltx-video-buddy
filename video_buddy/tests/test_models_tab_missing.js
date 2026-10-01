// Models tab: bundles[k].missing is {slot: filename}, not a list.
// node tests/test_models_tab_missing.js
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const htmlPath = path.join(__dirname, "..", "master_agent", "web", "static", "index.html");
const html = fs.readFileSync(htmlPath, "utf8");
const start = html.indexOf("function formatMissing");
const end = html.indexOf("async function loadModels");
if (start < 0 || end < start) {
  throw new Error("formatMissing must sit directly above loadModels");
}
const formatMissing = vm.runInNewContext(html.slice(start, end) + "\nformatMissing");
const got = formatMissing({ diffusion: "a.gguf", text_encoder: "gemma.safetensors" });
if (got !== "diffusion: a.gguf, text_encoder: gemma.safetensors") {
  throw new Error("object missing render: " + got);
}
if (formatMissing(["one.gguf", "two.safetensors"]) !== "one.gguf, two.safetensors") {
  throw new Error("array missing render failed");
}
if (formatMissing(null) !== "" || formatMissing(undefined) !== "") {
  throw new Error("empty missing render failed");
}

const loadEnd = html.indexOf("function bindDial");
const load = html.slice(end, loadEnd);
if (!load.includes("formatMissing")) throw new Error("loadModels must call formatMissing");
if (!load.includes("catch")) throw new Error("loadModels must catch render errors");
if (load.includes("(v.missing || []).join")) throw new Error("object join regression");
if (!html.includes("Clear used VRAM-min and continue a budget-paused generate")) {
  throw new Error("Reset used title missing");
}
console.log("ok");
