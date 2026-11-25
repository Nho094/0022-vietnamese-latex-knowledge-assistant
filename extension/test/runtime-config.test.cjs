const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const {
  modelRuntimeEnvironment,
  resolveModelRoot,
  serviceStateFromCapabilities,
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

test("backend is ready only when both model components are available", () => {
  assert.equal(serviceStateFromCapabilities(true, true), "ready");
  assert.equal(serviceStateFromCapabilities(true, false), "degraded");
  assert.equal(serviceStateFromCapabilities(false, true), "degraded");
  assert.equal(serviceStateFromCapabilities(false, false), "degraded");
});
