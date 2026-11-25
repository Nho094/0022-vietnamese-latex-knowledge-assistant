const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");
const {
  modelRuntimeEnvironment,
  pythonCandidates,
  resolveModelRoot,
  serviceStateFromCapabilities,
  syncSeededIntentsRegistry,
} = require("../dist/services/runtime-config.js");

test("prefers explicit modelRoot and resolves workspace default", () => {
  assert.equal(resolveModelRoot("/models/custom", "/work", "/storage"), "/models/custom");
  assert.equal(resolveModelRoot("models-alt", "/work/project", "/storage"), path.resolve("/work/project/models-alt"));
  assert.equal(resolveModelRoot("", "/work/project", "/storage"), path.join("/work/project", "models"));
  assert.equal(resolveModelRoot("", undefined, "/storage/profile"), path.join("/storage/profile", "models"));
});

test("passes offline Hugging Face cache locations to the backend", () => {
  assert.deepEqual(modelRuntimeEnvironment("/work/models"), {
    KA_MODEL_ROOT: "/work/models",
    KA_HF_CACHE_DIR: "/work/models/hf_cache",
    KA_PHOBERT_DIR: "/work/models/hf_cache/phobert-base-v2",
    KA_SBERT_DIR: "/work/models/hf_cache/vietnamese-bi-encoder",
    KA_CHECKPOINTS_DIR: "/work/models/checkpoints",
    KA_CHECKPOINT_DIR: "/work/models/checkpoints/phobert_classifier",
    KA_INDEX_CACHE_DIR: "/work/models",
    HF_HUB_OFFLINE: "1",
    TRANSFORMERS_OFFLINE: "1",
  });
});

test("backend state reflects capabilities and dependency errors accurately", () => {
  assert.equal(serviceStateFromCapabilities(true, true), "ready");
  assert.equal(serviceStateFromCapabilities(true, false), "degraded");
  assert.equal(serviceStateFromCapabilities(false, true), "degraded");
  assert.equal(serviceStateFromCapabilities(false, false), "degraded");
  assert.equal(serviceStateFromCapabilities(true, true, "error", true), "error");
  assert.equal(serviceStateFromCapabilities(false, false, "error"), "error");
});

test("searches workspace python venv before packaged extension path and PATH", () => {
  const candidates = pythonCandidates("", ["/workspace/project"], "/vsix/ext", "");
  assert.deepEqual(candidates.slice(0, 4), [
    path.join("/workspace/project", "AI", "bin", "python"),
    path.join("/workspace/project", ".venv", "bin", "python"),
    path.join("/vsix/ext", "AI", "bin", "python"),
    path.join("/vsix/ext", ".venv", "bin", "python"),
  ]);
  assert.ok(candidates.includes("python3"));

  const relativeConfigured = pythonCandidates("AI/bin/python", ["/workspace/project"], "/vsix/ext", "");
  assert.equal(relativeConfigured[0], path.resolve("/workspace/project", "AI/bin/python"));
});

test("upgrades unmodified seeded intents.json while preserving user edits", async () => {
  const tmp = await fs.mkdtemp(path.join(os.tmpdir(), "vka-seed-"));
  const bundled = path.join(tmp, "bundled.json");
  const registry = path.join(tmp, "registry.json");
  const marker = path.join(tmp, ".seeded.sha256");

  await fs.writeFile(bundled, '{"v":1}', "utf8");
  assert.equal(await syncSeededIntentsRegistry(bundled, registry, marker), true);
  assert.equal(await fs.readFile(registry, "utf8"), '{"v":1}');

  // Unmodified seeded file upgrades cleanly on bundle update
  await fs.writeFile(bundled, '{"v":2}', "utf8");
  assert.equal(await syncSeededIntentsRegistry(bundled, registry, marker), true);
  assert.equal(await fs.readFile(registry, "utf8"), '{"v":2}');

  // User-modified file is preserved across bundle updates
  await fs.writeFile(registry, '{"v":"custom"}', "utf8");
  await fs.writeFile(bundled, '{"v":3}', "utf8");
  assert.equal(await syncSeededIntentsRegistry(bundled, registry, marker), false);
  assert.equal(await fs.readFile(registry, "utf8"), '{"v":"custom"}');
});

test("upgrades legacy pre-marker stale intents.json and backs up previous file while keeping user custom arrays", async () => {
  const tmp = await fs.mkdtemp(path.join(os.tmpdir(), "vka-legacy-seed-"));
  const bundled = path.join(tmp, "bundled.json");
  const registry = path.join(tmp, "registry.json");
  const marker = path.join(tmp, ".seeded.sha256");

  const makeItems = (emptyLatex) => Array.from({ length: 12 }, (_, idx) => ({
    intent: `MATH_ITEM_${idx}`,
    category: "math_shared",
    templates: { latex: emptyLatex ? "" : `\\frac{1}{${idx + 1}}`, mathjax: `$$${idx}$$` },
  }));

  await fs.writeFile(registry, JSON.stringify(makeItems(true)), "utf8");
  await fs.writeFile(bundled, JSON.stringify(makeItems(false)), "utf8");

  assert.equal(await syncSeededIntentsRegistry(bundled, registry, marker), true);
  assert.equal(JSON.parse(await fs.readFile(registry, "utf8"))[0].templates.latex, "\\frac{1}{1}");
  assert.ok(await fs.readFile(`${registry}.pre-upgrade.bak`, "utf8"));

  // Pre-marker file with user custom intent MUST NOT be overwritten
  const marker2 = path.join(tmp, ".seeded2.sha256");
  const custom = [...makeItems(true), { intent: "USER_CUSTOM_RULE", category: "obsidian", templates: { obsidian: "# Hi" } }];
  await fs.writeFile(registry, JSON.stringify(custom), "utf8");
  assert.equal(await syncSeededIntentsRegistry(bundled, registry, marker2), false);
  assert.equal(JSON.parse(await fs.readFile(registry, "utf8")).length, 13);
});
