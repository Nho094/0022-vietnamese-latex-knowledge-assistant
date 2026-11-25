import * as path from "node:path";
import { homedir } from "node:os";

export function resolveModelRoot(
  configured: string,
  workspaceRoot: string | undefined,
  globalStorageRoot: string,
): string {
  const value = configured.trim();
  if (value) {
    const expanded = value.replace(/^~(?=$|[\\/])/, homedir());
    return path.resolve(workspaceRoot ?? globalStorageRoot, expanded);
  }
  return path.join(workspaceRoot ?? globalStorageRoot, "models");
}

export function modelRuntimeEnvironment(modelRoot: string): Record<string, string> {
  const hfCache = path.join(modelRoot, "hf_cache");
  return {
    KA_MODEL_ROOT: modelRoot,
    KA_HF_CACHE_DIR: hfCache,
    KA_PHOBERT_DIR: path.join(hfCache, "phobert-base-v2"),
    KA_SBERT_DIR: path.join(hfCache, "vietnamese-bi-encoder"),
    KA_CHECKPOINTS_DIR: path.join(modelRoot, "checkpoints"),
    KA_CHECKPOINT_DIR: path.join(modelRoot, "checkpoints", "phobert_classifier"),
    KA_INDEX_CACHE_DIR: modelRoot,
    HF_HUB_OFFLINE: "1",
    TRANSFORMERS_OFFLINE: "1",
  };
}

export function serviceStateFromCapabilities(classifierAvailable: boolean, retrievalModelAvailable: boolean): "ready" | "degraded" {
  return classifierAvailable && retrievalModelAvailable ? "ready" : "degraded";
}
