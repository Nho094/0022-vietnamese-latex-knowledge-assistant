import { randomBytes } from "node:crypto";
import * as fs from "node:fs/promises";
import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import { createServer } from "node:net";
import * as path from "node:path";
import * as vscode from "vscode";
import {
  modelRuntimeEnvironment,
  pythonCandidates,
  resolveModelRoot,
  serviceStateFromCapabilities,
  syncSeededIntentsRegistry,
} from "./runtime-config";

export type ServiceState = "starting" | "ready" | "degraded" | "missing" | "error" | "stopped";
export type ServiceStatus = {
  state: ServiceState;
  message: string;
  python?: string;
  pipelineMode?: string;
  classifierAvailable?: boolean;
  classifierMissingDependencies?: string[];
  classifierMessage?: string;
  classifierCheckpointPath?: string;
  phobertBasePath?: string;
  retrievalModelAvailable?: boolean;
  retrievalMissingDependencies?: string[];
  retrievalModelMessage?: string;
  retrievalModelPath?: string;
  retrievalIndexAvailable?: boolean;
  retrievalIndexMessage?: string;
  retrievalIndexPath?: string;
  modelRoot?: string;
  retrievalCacheAvailable?: boolean;
};

export class BackendService {
  private child: ChildProcess | undefined;
  private port: number | undefined;
  private sessionToken: string | undefined;
  private startup: Promise<ServiceStatus> | undefined;
  private lastOutput = "";
  private disposed = false;
  private status: ServiceStatus = { state: "stopped", message: "Dịch vụ chưa khởi động." };

  constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly output: vscode.OutputChannel,
    private readonly onStatus: (status: ServiceStatus) => void,
  ) {}

  get currentStatus(): ServiceStatus {
    return this.status;
  }

  async start(): Promise<ServiceStatus> {
    if (
      this.port &&
      this.child &&
      this.child.exitCode === null &&
      (this.status.state === "ready" || this.status.state === "degraded" || this.status.state === "error")
    ) {
      return this.status;
    }
    if (this.startup) return this.startup;
    this.disposed = false;
    this.setStatus({ state: "starting", message: "Đang khởi động dịch vụ kiến thức cục bộ…" });
    this.startup = this.startInternal().catch((error: unknown) => {
      const message = error instanceof Error ? error.message : "Lỗi khởi động không xác định.";
      const status: ServiceStatus = { state: "error", message: `Không chuẩn bị được dữ liệu cục bộ: ${message}` };
      this.setStatus(status);
      return status;
    }).finally(() => { this.startup = undefined; });
    return this.startup;
  }

  private async startInternal(): Promise<ServiceStatus> {
    const python = this.findPython();
    if (!python) {
      const status = {
        state: "missing" as const,
        message: "Không tìm thấy Python có FastAPI, Uvicorn và thư viện xử lý kiến thức. Chọn môi trường Python phù hợp trong Settings.",
      };
      this.setStatus(status);
      return status;
    }

    this.port = await reserveLocalPort();
    this.sessionToken = randomBytes(32).toString("hex");
    const root = this.context.extensionPath;
    const modelRoot = this.modelRoot();
    const storage = this.context.globalStorageUri.fsPath;
    const registryPath = path.join(storage, "intents.json");
    const seedMarkerPath = path.join(storage, ".seeded-intents.sha256");
    await fs.mkdir(storage, { recursive: true });
    await syncSeededIntentsRegistry(path.join(root, "data", "intents.json"), registryPath, seedMarkerPath);
    const env = {
      ...process.env,
      PYTHONPATH: [root, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
      KA_HOST: "127.0.0.1",
      KA_PORT: String(this.port),
      KA_SESSION_TOKEN: this.sessionToken,
      KA_INTENTS_PATH: registryPath,
      KA_BUILDER_MODEL: "",
      ...modelRuntimeEnvironment(modelRoot),
    };
    this.child = spawn(python, ["-m", "uvicorn", "api.server:app", "--host", "127.0.0.1", "--port", String(this.port)], {
      cwd: root,
      env,
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    });
    this.child.stdout?.on("data", (chunk: Buffer) => this.capture(chunk.toString()));
    this.child.stderr?.on("data", (chunk: Buffer) => this.capture(chunk.toString()));
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
      .get<number>("backendStartupTimeout", 25) * 1000;
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      if (!this.child || this.child.exitCode !== null) break;
      try {
        await this.fetchJson("/healthz", false);
        const info = await this.fetchJson<{
          status: string;
          message: string;
          python: string;
          pipeline_mode?: string;
          classifier_available: boolean;
          classifier_missing_dependencies?: string[];
          classifier_message: string;
          classifier_checkpoint_path: string;
          phobert_base_path: string;
          retrieval_model_available: boolean;
          retrieval_missing_dependencies?: string[];
          retrieval_model_message: string;
          retrieval_model_path: string;
          retrieval_index_available: boolean;
          retrieval_index_message: string;
          retrieval_index_path: string;
          model_root: string;
          retrieval_cache_available: boolean;
        }>("/system/status", true);
        const hasDependencyError =
          (info.classifier_missing_dependencies?.length ?? 0) > 0 ||
          (info.retrieval_missing_dependencies?.length ?? 0) > 0;
        const state = serviceStateFromCapabilities(
          info.classifier_available,
          info.retrieval_model_available,
          info.status,
          hasDependencyError,
        );
        const status: ServiceStatus = {
          state,
          message: info.message,
          python: info.python,
          pipelineMode: info.pipeline_mode,
          classifierAvailable: info.classifier_available,
          classifierMissingDependencies: info.classifier_missing_dependencies,
          classifierMessage: info.classifier_message,
          classifierCheckpointPath: info.classifier_checkpoint_path,
          phobertBasePath: info.phobert_base_path,
          retrievalModelAvailable: info.retrieval_model_available,
          retrievalMissingDependencies: info.retrieval_missing_dependencies,
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
      } catch {
        await delay(250);
      }
    }
    const logs = this.lastOutput.trim().slice(-900);
    const message = logs
      ? `Không thể khởi động dịch vụ. ${logs}`
      : "Hết thời gian chờ khởi động dịch vụ. Kiểm tra Python và các thư viện backend.";
    const status: ServiceStatus = { state: "error", message };
    this.setStatus(status);
    await this.stop();
    return status;
  }

  async request<T>(route: string, options: { method?: string; body?: unknown } = {}): Promise<T> {
    if (!this.port || !this.child || this.child.exitCode !== null || this.status.state === "stopped" || this.status.state === "missing") {
      const started = await this.start();
      if (!this.port || started.state === "missing" || started.state === "stopped") {
        throw new Error(started.message);
      }
    }
    return this.fetchJson<T>(route, true, options);
  }

  async stop(): Promise<void> {
    this.disposed = true;
    const processToStop = this.child;
    this.child = undefined;
    this.port = undefined;
    this.sessionToken = undefined;
    if (processToStop && processToStop.exitCode === null) {
      processToStop.kill();
      await Promise.race([
        new Promise<void>((resolve) => processToStop.once("close", () => resolve())),
        delay(2000),
      ]);
      if (processToStop.exitCode === null) processToStop.kill("SIGKILL");
    }
    this.setStatus({ state: "stopped", message: "Dịch vụ cục bộ đã dừng." });
  }

  private async fetchJson<T>(route: string, authenticated: boolean, options: { method?: string; body?: unknown } = {}): Promise<T> {
    if (!this.port) throw new Error("Dịch vụ backend chưa sẵn sàng.");
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
      const value = await response.json() as T & { detail?: unknown };
      if (!response.ok) {
        const message = typeof value.detail === "string" ? value.detail : JSON.stringify(value.detail ?? value);
        throw new Error(message || `Backend trả mã ${response.status}.`);
      }
      return value;
    } finally {
      clearTimeout(timeout);
    }
  }

  private findPython(): string | undefined {
    const configured = vscode.workspace.getConfiguration("vietnameseKnowledgeAssistant").get<string>("pythonPath", "").trim();
    const root = this.context.extensionPath;
    const workspaceRoots = (vscode.workspace.workspaceFolders ?? []).map((folder) => folder.uri.fsPath);
    const candidates = pythonCandidates(configured, workspaceRoots, root);
    let fallbackCandidate: string | undefined;
    for (const candidate of candidates) {
      const check = spawnSync(
        candidate,
        [
          "-c",
          "import importlib.util; " +
            "req = ('fastapi', 'uvicorn', 'jsonschema', 'jsonata', 'jinja2'); " +
            "ok = all(importlib.util.find_spec(m) is not None for m in req); " +
            "ml = any(importlib.util.find_spec(m) is not None for m in ('torch', 'sentence_transformers')); " +
            "print(('ready_ml' if ml else 'ready_base') if ok else 'missing_deps')",
        ],
        {
          cwd: workspaceRoots[0] ?? root,
          encoding: "utf8",
          timeout: 15_000,
          windowsHide: true,
        },
      );
      if (check.status === 0 && check.stdout.includes("ready_ml")) {
        this.output.appendLine(`[Python] Sử dụng môi trường có ML: ${candidate}`);
        return candidate;
      }
      if (check.status === 0 && check.stdout.includes("ready_base") && !fallbackCandidate) {
        this.output.appendLine(`[Python] Tìm thấy môi trường cơ bản (thiếu ML): ${candidate}`);
        fallbackCandidate = candidate;
        continue;
      }
      const detail = check.error?.message ?? (check.stderr || check.stdout || `mã ${check.status}`).trim();
      this.output.appendLine(`[Python] Bỏ qua '${candidate}': ${detail}`);
    }
    return fallbackCandidate;
  }

  private modelRoot(): string {
    const configured = vscode.workspace.getConfiguration("vietnameseKnowledgeAssistant").get<string>("modelRoot", "").trim();
    const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    return resolveModelRoot(configured, workspaceRoot, this.context.globalStorageUri.fsPath);
  }

  private capture(value: string): void {
    this.lastOutput = (this.lastOutput + value).slice(-8000);
    this.output.append(value);
  }

  private setStatus(status: ServiceStatus): void {
    this.status = status;
    this.onStatus(status);
  }
}

async function reserveLocalPort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = createServer();
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

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
