"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.resolveModelRoot = resolveModelRoot;
exports.pythonCandidates = pythonCandidates;
exports.syncSeededIntentsRegistry = syncSeededIntentsRegistry;
exports.modelRuntimeEnvironment = modelRuntimeEnvironment;
exports.serviceStateFromCapabilities = serviceStateFromCapabilities;
const node_crypto_1 = require("node:crypto");
const fs = __importStar(require("node:fs/promises"));
const path = __importStar(require("node:path"));
const node_os_1 = require("node:os");
function resolveModelRoot(configured, workspaceRoot, globalStorageRoot) {
    const value = configured.trim();
    if (value) {
        const expanded = value.replace(/^~(?=$|[\\/])/, (0, node_os_1.homedir)());
        return path.resolve(workspaceRoot ?? globalStorageRoot, expanded);
    }
    return path.join(workspaceRoot ?? globalStorageRoot, "models");
}
function pythonCandidates(configured, workspaceRoots, extensionPath, virtualEnvRoot = process.env.VIRTUAL_ENV || process.env.CONDA_PREFIX) {
    const cleanedConfig = configured.trim().replace(/^['"]|['"]$/g, "");
    const expandedConfig = [];
    if (cleanedConfig) {
        const withHome = cleanedConfig.replace(/^~(?=$|[\\/])/, (0, node_os_1.homedir)());
        if (path.isAbsolute(withHome) || !/[\\/]/.test(withHome)) {
            expandedConfig.push(withHome);
        }
        else if (workspaceRoots[0]) {
            expandedConfig.push(path.resolve(workspaceRoots[0], withHome));
            expandedConfig.push(withHome);
        }
        else {
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
function hasLegacyPreMarkerSeedSignature(existingContent, bundledContent) {
    try {
        const existingParsed = JSON.parse(existingContent);
        const bundledParsed = JSON.parse(bundledContent);
        if (!Array.isArray(existingParsed) || !Array.isArray(bundledParsed) || existingParsed.length !== bundledParsed.length) {
            return false;
        }
        const bundledIds = new Set(bundledParsed
            .filter((item) => Boolean(item && typeof item === "object" && typeof item.intent === "string"))
            .map((item) => item.intent));
        if (bundledIds.size !== bundledParsed.length)
            return false;
        let legacyEmptyLatexMathShared = 0;
        let legacyPlaceholderExamples = 0;
        for (const raw of existingParsed) {
            if (!raw || typeof raw !== "object")
                return false;
            const item = raw;
            if (typeof item.intent !== "string" || !bundledIds.has(item.intent))
                return false;
            if (item.assistant_status === "paused" || item.custom === true)
                return false;
            if (item.category === "math_shared") {
                const templates = item.templates;
                if (templates && templates.latex === "")
                    legacyEmptyLatexMathShared += 1;
            }
            const slots = item.slots;
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
    }
    catch {
        return false;
    }
}
async function syncSeededIntentsRegistry(bundledPath, registryPath, markerPath) {
    const bundledContent = await fs.readFile(bundledPath, "utf8");
    const bundledHash = (0, node_crypto_1.createHash)("sha256").update(bundledContent, "utf8").digest("hex");
    let existingContent;
    try {
        existingContent = await fs.readFile(registryPath, "utf8");
    }
    catch (error) {
        if (!(error instanceof Error && "code" in error && error.code === "ENOENT"))
            throw error;
    }
    if (existingContent === undefined) {
        await fs.writeFile(registryPath, bundledContent, "utf8");
        await fs.writeFile(markerPath, bundledHash + "\n", "utf8");
        return true;
    }
    const currentHash = (0, node_crypto_1.createHash)("sha256").update(existingContent, "utf8").digest("hex");
    if (currentHash === bundledHash) {
        await fs.writeFile(markerPath, bundledHash + "\n", "utf8");
        return false;
    }
    let lastSeedHash = "";
    try {
        lastSeedHash = (await fs.readFile(markerPath, "utf8")).trim();
    }
    catch (error) {
        if (!(error instanceof Error && "code" in error && error.code === "ENOENT"))
            throw error;
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
function modelRuntimeEnvironment(modelRoot) {
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
function serviceStateFromCapabilities(classifierAvailable, retrievalModelAvailable, backendStatus, hasDependencyError) {
    if (hasDependencyError || backendStatus === "error") {
        return "error";
    }
    return classifierAvailable && retrievalModelAvailable ? "ready" : "degraded";
}
//# sourceMappingURL=runtime-config.js.map