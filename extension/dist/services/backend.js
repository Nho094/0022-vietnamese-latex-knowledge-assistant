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
exports.BackendService = void 0;
const node_crypto_1 = require("node:crypto");
const node_fs_1 = require("node:fs");
const fs = __importStar(require("node:fs/promises"));
const node_child_process_1 = require("node:child_process");
const node_net_1 = require("node:net");
const path = __importStar(require("node:path"));
const vscode = __importStar(require("vscode"));
const runtime_config_1 = require("./runtime-config");
class BackendService {
    context;
    output;
    onStatus;
    child;
    port;
    sessionToken;
    startup;
    lastOutput = "";
    disposed = false;
    status = { state: "stopped", message: "Dịch vụ chưa khởi động." };
    constructor(context, output, onStatus) {
        this.context = context;
        this.output = output;
        this.onStatus = onStatus;
    }
    get currentStatus() {
        return this.status;
    }
    async start() {
        if (this.status.state === "ready" || this.status.state === "degraded")
            return this.status;
        if (this.startup)
            return this.startup;
        this.disposed = false;
        this.setStatus({ state: "starting", message: "Đang khởi động dịch vụ kiến thức cục bộ…" });
        this.startup = this.startInternal().catch((error) => {
            const message = error instanceof Error ? error.message : "Lỗi khởi động không xác định.";
            const status = { state: "error", message: `Không chuẩn bị được dữ liệu cục bộ: ${message}` };
            this.setStatus(status);
            return status;
        }).finally(() => { this.startup = undefined; });
        return this.startup;
    }
    async startInternal() {
        const python = this.findPython();
        if (!python) {
            const status = {
                state: "missing",
                message: "Không tìm thấy Python có FastAPI, Uvicorn và thư viện xử lý kiến thức. Chọn môi trường Python phù hợp trong Settings.",
            };
            this.setStatus(status);
            return status;
        }
        this.port = await reserveLocalPort();
        this.sessionToken = (0, node_crypto_1.randomBytes)(32).toString("hex");
        const root = this.context.extensionPath;
        const modelRoot = this.modelRoot();
        const storage = this.context.globalStorageUri.fsPath;
        const registryPath = path.join(storage, "intents.json");
        await fs.mkdir(storage, { recursive: true });
        try {
            await fs.copyFile(path.join(root, "data", "intents.json"), registryPath, node_fs_1.constants.COPYFILE_EXCL);
        }
        catch (error) {
            if (!(error instanceof Error && "code" in error && error.code === "EEXIST"))
                throw error;
        }
        const env = {
            ...process.env,
            PYTHONPATH: [root, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
            KA_HOST: "127.0.0.1",
            KA_PORT: String(this.port),
            KA_SESSION_TOKEN: this.sessionToken,
            KA_INTENTS_PATH: registryPath,
            KA_BUILDER_MODEL: "",
            ...(0, runtime_config_1.modelRuntimeEnvironment)(modelRoot),
        };
        this.child = (0, node_child_process_1.spawn)(python, ["-m", "uvicorn", "api.server:app", "--host", "127.0.0.1", "--port", String(this.port)], {
            cwd: root,
            env,
            stdio: ["ignore", "pipe", "pipe"],
            windowsHide: true,
        });
        this.child.stdout?.on("data", (chunk) => this.capture(chunk.toString()));
        this.child.stderr?.on("data", (chunk) => this.capture(chunk.toString()));
        this.child.on("error", (error) => {
            this.capture(error.message);
            this.setStatus({ state: "error", message: `Không khởi chạy được dịch vụ: ${error.message}` });
        });
        this.child.on("close", (code) => {
            if (!this.disposed) {
                const details = this.lastOutput.trim().slice(-700);
                this.setStatus({
                    state: "error",
                    message: `Dịch vụ đã dừng${code === null ? "" : ` (mã ${code})`}.${details ? ` ${details}` : ""}`,
                });
            }
            this.child = undefined;
        });
        const timeoutMs = vscode.workspace.getConfiguration("vietnameseKnowledgeAssistant")
            .get("backendStartupTimeout", 25) * 1000;
        const deadline = Date.now() + timeoutMs;
        while (Date.now() < deadline) {
            if (!this.child || this.child.exitCode !== null)
                break;
            try {
                await this.fetchJson("/healthz", false);
                const info = await this.fetchJson("/system/status", true);
                const state = (0, runtime_config_1.serviceStateFromCapabilities)(info.classifier_available, info.retrieval_model_available);
                const status = {
                    state,
                    message: info.message,
                    python: info.python,
                    classifierAvailable: info.classifier_available,
                    classifierMessage: info.classifier_message,
                    classifierCheckpointPath: info.classifier_checkpoint_path,
                    phobertBasePath: info.phobert_base_path,
                    retrievalModelAvailable: info.retrieval_model_available,
                    retrievalModelMessage: info.retrieval_model_message,
                    retrievalModelPath: info.retrieval_model_path,
                    retrievalIndexAvailable: info.retrieval_index_available,
                    retrievalIndexMessage: info.retrieval_index_message,
                    retrievalIndexPath: info.retrieval_index_path,
                    modelRoot: info.model_root,
                    retrievalCacheAvailable: info.retrieval_cache_available,
                };
                this.setStatus(status);
                return status;
            }
            catch {
                await delay(250);
            }
        }
        const logs = this.lastOutput.trim().slice(-900);
        const message = logs
            ? `Không thể khởi động dịch vụ. ${logs}`
            : "Hết thời gian chờ khởi động dịch vụ. Kiểm tra Python và các thư viện backend.";
        const status = { state: "error", message };
        this.setStatus(status);
        await this.stop();
        return status;
    }
    async request(route, options = {}) {
        if (this.status.state === "stopped" || this.status.state === "missing" || this.status.state === "error") {
            const started = await this.start();
            if (started.state !== "ready" && started.state !== "degraded")
                throw new Error(started.message);
        }
        return this.fetchJson(route, true, options);
    }
    async stop() {
        this.disposed = true;
        const processToStop = this.child;
        this.child = undefined;
        this.port = undefined;
        this.sessionToken = undefined;
        if (processToStop && processToStop.exitCode === null) {
            processToStop.kill();
            await Promise.race([
                new Promise((resolve) => processToStop.once("close", () => resolve())),
                delay(2000),
            ]);
            if (processToStop.exitCode === null)
                processToStop.kill("SIGKILL");
        }
        this.setStatus({ state: "stopped", message: "Dịch vụ cục bộ đã dừng." });
    }
    async fetchJson(route, authenticated, options = {}) {
        if (!this.port)
            throw new Error("Dịch vụ backend chưa sẵn sàng.");
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 15_000);
        try {
            const response = await fetch(`http://127.0.0.1:${this.port}${route}`, {
                method: options.method ?? "GET",
                headers: {
                    ...(options.body === undefined ? {} : { "Content-Type": "application/json" }),
                    ...(authenticated && this.sessionToken ? { "X-KA-Session": this.sessionToken } : {}),
                },
                body: options.body === undefined ? undefined : JSON.stringify(options.body),
                signal: controller.signal,
            });
            const value = await response.json();
            if (!response.ok) {
                const message = typeof value.detail === "string" ? value.detail : JSON.stringify(value.detail ?? value);
                throw new Error(message || `Backend trả mã ${response.status}.`);
            }
            return value;
        }
        finally {
            clearTimeout(timeout);
        }
    }
    findPython() {
        const configured = vscode.workspace.getConfiguration("vietnameseKnowledgeAssistant").get("pythonPath", "").trim();
        const root = this.context.extensionPath;
        const candidates = [
            configured,
            path.join(root, "AI", "bin", "python"),
            path.join(root, ".venv", "bin", "python"),
            "python3",
            "python",
        ].filter(Boolean);
        for (const candidate of candidates) {
            const check = (0, node_child_process_1.spawnSync)(candidate, ["-c", "import fastapi, uvicorn, jsonschema, jsonata, jinja2; print('ready')"], {
                cwd: root,
                encoding: "utf8",
                timeout: 8000,
                windowsHide: true,
            });
            if (check.status === 0 && check.stdout.includes("ready"))
                return candidate;
            if (configured && candidate === configured && check.error) {
                this.output.appendLine(`Không dùng được Python đã cấu hình: ${check.error.message}`);
            }
        }
        return undefined;
    }
    modelRoot() {
        const configured = vscode.workspace.getConfiguration("vietnameseKnowledgeAssistant").get("modelRoot", "").trim();
        const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
        return (0, runtime_config_1.resolveModelRoot)(configured, workspaceRoot, this.context.globalStorageUri.fsPath);
    }
    capture(value) {
        this.lastOutput = (this.lastOutput + value).slice(-8000);
        this.output.append(value);
    }
    setStatus(status) {
        this.status = status;
        this.onStatus(status);
    }
}
exports.BackendService = BackendService;
async function reserveLocalPort() {
    return new Promise((resolve, reject) => {
        const server = (0, node_net_1.createServer)();
        server.once("error", reject);
        server.listen(0, "127.0.0.1", () => {
            const address = server.address();
            if (!address || typeof address === "string") {
                server.close();
                reject(new Error("Không cấp được cổng loopback."));
                return;
            }
            server.close((error) => error ? reject(error) : resolve(address.port));
        });
    });
}
function delay(milliseconds) {
    return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
//# sourceMappingURL=backend.js.map