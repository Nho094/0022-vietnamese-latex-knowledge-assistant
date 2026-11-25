import { createHash } from "node:crypto";
import * as fs from "node:fs/promises";
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

export function pythonCandidates(
  configured: string,
  workspaceRoots: readonly string[],
  extensionPath: string,
  virtualEnvRoot: string | undefined = process.env.VIRTUAL_ENV || process.env.CONDA_PREFIX,
): string[] {
  const cleanedConfig = configured.trim().replace(/^['"]|['"]$/g, "");
  const expandedConfig: string[] = [];
  if (cleanedConfig) {
    const withHome = cleanedConfig.replace(/^~(?=$|[\\/])/, homedir());
    if (path.isAbsolute(withHome) || !/[\\/]/.test(withHome)) {
      expandedConfig.push(withHome);
    } else if (workspaceRoots[0]) {
      expandedConfig.push(path.resolve(workspaceRoots[0], withHome));
      expandedConfig.push(withHome);
    } else {
      expandedConfig.push(withHome);
    }
  }
  const activeVenv = virtualEnvRoot?.trim()
    ? [path.join(virtualEnvRoot.trim(), "bin", "python")]
    : [];
  const raw = [
    ...expandedConfig,
    ...workspaceRoots.flatMap((root) => [
      path.join(root, "AI", "bin", "python"),
      path.join(root, ".venv", "bin", "python"),
    ]),
    ...activeVenv,
    path.join(extensionPath, "AI", "bin", "python"),
    path.join(extensionPath, ".venv", "bin", "python"),
    "python3",
    "python",
  ].filter(Boolean);
  return Array.from(new Set(raw));
}

function hasLegacyPreMarkerSeedSignature(
  existingContent: string,
  bundledContent: string,
): boolean {
  try {
    const existingParsed = JSON.parse(existingContent);
    const bundledParsed = JSON.parse(bundledContent);
    if (!Array.isArray(existingParsed) || !Array.isArray(bundledParsed) || existingParsed.length !== bundledParsed.length) {
      return false;
    }
    const bundledIds = new Set(
      bundledParsed
        .filter((item): item is Record<string, unknown> => Boolean(item && typeof item === "object" && typeof (item as Record<string, unknown>).intent === "string"))
        .map((item) => item.intent as string),
    );
    if (bundledIds.size !== bundledParsed.length) return false;

    let legacyEmptyLatexMathShared = 0;
    let legacyPlaceholderExamples = 0;
    for (const raw of existingParsed) {
      if (!raw || typeof raw !== "object") return false;
      const item = raw as Record<string, unknown>;
      if (typeof item.intent !== "string" || !bundledIds.has(item.intent)) return false;
      if (item.assistant_status === "paused" || item.custom === true) return false;

      if (item.category === "math_shared") {
        const templates = item.templates as Record<string, unknown> | undefined;
        if (templates && templates.latex === "") legacyEmptyLatexMathShared += 1;
      }
      const slots = item.slots as Record<string, Record<string, unknown>> | undefined;
      if (slots && typeof slots === "object") {
        for (const sdef of Object.values(slots)) {
          const ex = sdef?.examples;
          if (Array.isArray(ex) && (ex.includes("noi dung mau") || ex.includes("noi dung thu nghiem"))) {
            legacyPlaceholderExamples += 1;
          }
        }
      }
    }
    return legacyEmptyLatexMathShared >= 10 || legacyPlaceholderExamples >= 10;
  } catch {
    return false;
  }
}

export async function syncSeededIntentsRegistry(
  bundledPath: string,
  registryPath: string,
  markerPath: string,
): Promise<boolean> {
  const bundledContent = await fs.readFile(bundledPath, "utf8");
  const bundledHash = createHash("sha256").update(bundledContent, "utf8").digest("hex");
  let existingContent: string | undefined;
  try {
    existingContent = await fs.readFile(registryPath, "utf8");
  } catch (error) {
    if (!(error instanceof Error && "code" in error && error.code === "ENOENT")) throw error;
  }

  if (existingContent === undefined) {
    await fs.writeFile(registryPath, bundledContent, "utf8");
    await fs.writeFile(markerPath, bundledHash + "\n", "utf8");
    return true;
  }

  const currentHash = createHash("sha256").update(existingContent, "utf8").digest("hex");
  if (currentHash === bundledHash) {
    await fs.writeFile(markerPath, bundledHash + "\n", "utf8");
    return false;
  }

  let lastSeedHash = "";
  try {
    lastSeedHash = (await fs.readFile(markerPath, "utf8")).trim();
  } catch (error) {
    if (!(error instanceof Error && "code" in error && error.code === "ENOENT")) throw error;
  }

  if (lastSeedHash && currentHash === lastSeedHash) {
    const tempPath = `${registryPath}.tmp`;
    await fs.writeFile(tempPath, bundledContent, "utf8");
    await fs.rename(tempPath, registryPath);
    await fs.writeFile(markerPath, bundledHash + "\n", "utf8");
    return true;
  }

  // Nâng cấp an toàn từ phiên bản Extension cũ chưa tạo tệp .seeded-intents.sha256
  if (!lastSeedHash && hasLegacyPreMarkerSeedSignature(existingContent, bundledContent)) {
    const backupPath = `${registryPath}.pre-upgrade.bak`;
    await fs.writeFile(backupPath, existingContent, "utf8");
    const tempPath = `${registryPath}.tmp`;
    await fs.writeFile(tempPath, bundledContent, "utf8");
    await fs.rename(tempPath, registryPath);
    await fs.writeFile(markerPath, bundledHash + "\n", "utf8");
    return true;
  }

  return false;
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

export function serviceStateFromCapabilities(
  classifierAvailable: boolean,
  retrievalModelAvailable: boolean,
  backendStatus?: string,
  hasDependencyError?: boolean,
): "ready" | "degraded" | "error" {
  if (hasDependencyError || backendStatus === "error") {
    return "error";
  }
  return classifierAvailable && retrievalModelAvailable ? "ready" : "degraded";
}
