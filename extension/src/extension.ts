import * as fs from "node:fs/promises";
import { randomBytes } from "node:crypto";
import * as path from "node:path";
import * as vscode from "vscode";
import { BackendService, type ServiceStatus } from "./services/backend";
import { directiveCommentPrefixes, parseKnowledgeDirective } from "./editor/directive";

type Definition = Record<string, unknown> & { intent: string; display_name?: string; describe?: string };
type Activity = { id: string; action: string; intent?: string; title: string; at: string };
type BridgeMessage = { id?: string; action?: string; payload?: Record<string, unknown> };
type PendingEditorTarget = { editor: vscode.TextEditor; range: vscode.Range; documentVersion: number };

const ACTIVITY_KEY = "vietnameseKnowledgeAssistant.activity";
const STATUS_KEY = "vietnameseKnowledgeAssistant.knowledgeStatus";
const THEME_KEY = "vietnameseKnowledgeAssistant.theme";

class KnowledgeExtension {
  private readonly backend: BackendService;
  private readonly output: vscode.OutputChannel;
  private panel: vscode.WebviewPanel | undefined;
  private pendingEditorTarget: PendingEditorTarget | undefined;
  private pendingEditorOutput: string | undefined;
  private initialPage = "dashboard";
  private readonly statusItem: vscode.StatusBarItem;

  constructor(private readonly context: vscode.ExtensionContext) {
    this.output = vscode.window.createOutputChannel("Vietnamese Knowledge Assistant");
    this.backend = new BackendService(context, this.output, (status) => this.onBackendStatus(status));
    this.statusItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 20);
    this.statusItem.command = "vietnameseKnowledgeAssistant.checkBackend";
    this.statusItem.show();
    context.subscriptions.push(this.output, this.statusItem);
  }

  activate(): void {
      this.context.subscriptions.push(
      vscode.commands.registerCommand("vietnameseKnowledgeAssistant.openDashboard", () => this.openPanel("dashboard")),
      vscode.commands.registerCommand("vietnameseKnowledgeAssistant.createKnowledge", () => this.openPanel("builder")),
      vscode.commands.registerCommand("vietnameseKnowledgeAssistant.checkBackend", () => this.checkBackend()),
      vscode.commands.registerCommand(
        "vietnameseKnowledgeAssistant.runDirective",
        (uri?: vscode.Uri, line?: number) => this.runDirective(uri, line),
      ),
      vscode.languages.registerCodeActionsProvider(
        Object.keys(directiveCommentPrefixes).map((language) => ({ language })),
        new DirectiveCodeActionProvider(),
        { providedCodeActionKinds: [vscode.CodeActionKind.QuickFix] },
      ),
      vscode.window.registerWebviewPanelSerializer("vietnameseKnowledgeAssistant.dashboard", {
        deserializeWebviewPanel: async (panel) => {
          this.initialPage = "dashboard";
          panel.webview.options = this.webviewOptions();
          this.attachPanel(panel);
        },
      }),
    );
    void this.backend.start();
  }

  async deactivate(): Promise<void> {
    await this.backend.stop();
  }

  private onBackendStatus(status: ServiceStatus): void {
    const labels: Record<ServiceStatus["state"], string> = {
      starting: "$(loading~spin) Trợ lý: đang khởi động",
      ready: "$(pass) Trợ lý: sẵn sàng",
      degraded: "$(warning) Trợ lý: chế độ hạn chế",
      missing: "$(error) Trợ lý: thiếu Python",
      error: "$(error) Trợ lý: lỗi dịch vụ",
      stopped: "$(circle-slash) Trợ lý: đã dừng",
    };
    this.statusItem.text = labels[status.state];
    this.statusItem.tooltip = status.message;
    this.panel?.webview.postMessage({ type: "serviceStatus", status });
  }

  private async checkBackend(): Promise<void> {
    const status = await this.backend.start();
    const action = status.state === "missing" || status.state === "error" ? "Mở cài đặt" : "Mở bảng điều khiển";
    const selected = await vscode.window.showInformationMessage(status.message, action, "Xem nhật ký");
    if (selected === "Mở cài đặt") {
      await vscode.commands.executeCommand("workbench.action.openSettings", "vietnameseKnowledgeAssistant.pythonPath");
    } else if (selected === "Mở bảng điều khiển") {
      this.openPanel("dashboard");
    } else if (selected === "Xem nhật ký") {
      this.output.show(true);
    }
  }

  private openPanel(page: string): void {
    this.initialPage = page;
    if (this.panel) {
      this.panel.reveal(vscode.ViewColumn.Beside);
      this.panel.webview.postMessage({ type: "navigate", page });
      return;
    }
    const panel = vscode.window.createWebviewPanel(
      "vietnameseKnowledgeAssistant.dashboard",
      "Vietnamese Knowledge Assistant",
      vscode.ViewColumn.Beside,
      { ...this.webviewOptions(), retainContextWhenHidden: true },
    );
    this.attachPanel(panel);
  }

  private webviewOptions(): vscode.WebviewOptions {
    return {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.context.extensionUri, "extension", "webview", "dist")],
    };
  }

  private attachPanel(panel: vscode.WebviewPanel): void {
    this.panel = panel;
    panel.webview.html = this.webviewHtml(panel.webview);
    panel.onDidDispose(() => { this.panel = undefined; }, null, this.context.subscriptions);
    panel.webview.onDidReceiveMessage((message: BridgeMessage) => void this.handleMessage(panel, message), null, this.context.subscriptions);
  }

  private webviewHtml(webview: vscode.Webview): string {
    const root = vscode.Uri.joinPath(this.context.extensionUri, "extension", "webview", "dist");
    const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(root, "assets", "app.js"));
    const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(root, "assets", "app.css"));
    const nonce = randomNonce();
    const csp = [
      "default-src 'none'",
      `img-src ${webview.cspSource} data:`,
      `font-src ${webview.cspSource} data:`,
      `style-src 'nonce-${nonce}' ${webview.cspSource}`,
      `script-src 'nonce-${nonce}' ${webview.cspSource}`,
      "connect-src 'none'",
    ].join("; ");
    return `<!doctype html>
<html lang="vi"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<meta http-equiv="Content-Security-Policy" content="${csp}"><title>Vietnamese Knowledge Assistant</title>
<link rel="stylesheet" href="${styleUri}"><style nonce="${nonce}">
html,body{margin:0;min-height:100%;background:#101014;color:#f4f4f5;font:14px system-ui,sans-serif}
#boot-status{margin:24px;padding:16px;border:1px solid #454550;border-radius:8px;white-space:pre-wrap}
</style></head><body><div id="root"></div><div id="boot-status">Đang tải giao diện Knowledge Assistant…</div>
<script nonce="${nonce}">
(() => {
  const show = (message) => { const node = document.getElementById('boot-status'); if (node) node.textContent = message; };
  window.addEventListener('error', (event) => {
    const target = event.target;
    if (target instanceof HTMLScriptElement || target instanceof HTMLLinkElement) {
      show('Không tải được tài nguyên giao diện: ' + (target.src || target.href || target.tagName));
    } else if (event.message) show('Lỗi JavaScript trong giao diện: ' + event.message);
  }, true);
  window.addEventListener('unhandledrejection', (event) => {
    const reason = event.reason;
    show('Lỗi khởi động giao diện: ' + (reason instanceof Error ? reason.message : String(reason)));
  });
})();
</script>
<script defer nonce="${nonce}" src="${scriptUri}"></script></body></html>`;
  }

  private async handleMessage(panel: vscode.WebviewPanel, message: BridgeMessage): Promise<void> {
    if (!message.action || !message.id) return;
    try {
      const payload = message.payload ?? {};
      const data = await this.handleAction(message.action, payload);
      panel.webview.postMessage({ type: "response", id: message.id, data });
    } catch (error) {
      panel.webview.postMessage({
        type: "response",
        id: message.id,
        error: error instanceof Error ? error.message : "Có lỗi không xác định.",
      });
    }
  }

  private async handleAction(action: string, payload: Record<string, unknown>): Promise<unknown> {
    switch (action) {
      case "bootstrap": {
        const [registry, corpus, history, system, builder] = await Promise.all([
          this.backend.request<{ definitions: Definition[]; categories: string[]; domains: string[] }>("/knowledge/definitions"),
          this.backend.request<Record<string, unknown>>("/knowledge/corpus"),
          this.backend.request<Record<string, unknown>>("/knowledge/history"),
          this.backend.request<Record<string, unknown>>("/system/status"),
          this.backend.request<Record<string, unknown>>("/knowledge/builder/capabilities"),
        ]);
        const statuses = this.context.globalState.get<Record<string, string>>(STATUS_KEY, {});
        return {
          ...registry,
          definitions: registry.definitions.map((definition) => ({
            ...definition,
            assistant_status: statuses[definition.intent] ?? "active",
          })),
          corpus,
          history,
          system,
          builder,
          activity: this.activities(),
          theme: this.context.globalState.get<string>(THEME_KEY, "dark"),
          initialPage: this.initialPage,
          editorPreview: this.pendingEditorOutput,
          service: this.backend.currentStatus,
        };
      }
      case "getDefinition":
        return this.backend.request(`/knowledge/definitions/${encodeURIComponent(String(payload.intent ?? ""))}`);
      case "compileBuilder":
        return this.backend.request("/knowledge/builder/compile", { method: "POST", body: payload });
      case "validateDefinition":
        return this.backend.request("/knowledge/validate", { method: "POST", body: { definition: payload.definition } });
      case "previewDefinition":
        return this.backend.request("/knowledge/preview", { method: "POST", body: payload });
      case "publishDefinition": {
        const overwrite = payload.overwrite === true;
        if (overwrite) {
          const confirm = await vscode.window.showWarningMessage(
            `Cập nhật “${String(payload.title ?? "kiến thức này") }”? Hệ thống sẽ lưu bản sao lưu trước khi thay đổi.`,
            { modal: true }, "Cập nhật có backup",
          );
          if (confirm !== "Cập nhật có backup") return { cancelled: true };
        }
        const result = await this.backend.request<{ published: boolean; intent: string }>("/knowledge/publish", {
          method: "POST",
          body: { definition: payload.definition, overwrite },
        });
        this.logActivity(overwrite ? "updated" : "created", result.intent, String(payload.title ?? result.intent));
        return result;
      }
      case "deleteDefinition": {
        const intent = String(payload.intent ?? "");
        const confirm = await vscode.window.showWarningMessage(
          `Xóa kiến thức “${String(payload.title ?? intent)}”? Có thể phục hồi từ lịch sử phiên bản.`,
          { modal: true }, "Xóa kiến thức",
        );
        if (confirm !== "Xóa kiến thức") return { cancelled: true };
        const result = await this.backend.request<{ deleted: boolean; intent: string }>(
          `/knowledge/definitions/${encodeURIComponent(intent)}`, { method: "DELETE" },
        );
        this.logActivity("deleted", intent, String(payload.title ?? intent));
        return result;
      }
      case "toggleKnowledge": {
        const intent = String(payload.intent ?? "");
        const statuses = this.context.globalState.get<Record<string, string>>(STATUS_KEY, {});
        const next = payload.enabled === false ? "paused" : "active";
        await this.context.globalState.update(STATUS_KEY, { ...statuses, [intent]: next });
        this.logActivity(next === "active" ? "activated" : "paused", intent, String(payload.title ?? intent));
        return { intent, status: next };
      }
      case "ask": {
        const result = await this.backend.request<Record<string, any>>("/ask", {
          method: "POST",
          body: { text: String(payload.text ?? ""), target: String(payload.target ?? "latex") },
        });
        return this.filterDisabled(result);
      }
      case "choose": {
        const result = await this.backend.request<Record<string, any>>("/ask/choose", {
          method: "POST",
          body: {
            text: String(payload.text ?? ""),
            candidate: String(payload.candidate ?? ""),
            values: payload.values ?? {},
            target: String(payload.target ?? "latex"),
          },
        });
        return this.filterDisabled(result);
      }
      case "rollback": {
        const version = String(payload.version ?? "");
        const confirm = await vscode.window.showWarningMessage(
          "Phục hồi sẽ thay bank kiến thức hiện tại bằng snapshot đã chọn. Extension tạo backup trước khi thay đổi.",
          { modal: true }, "Phục hồi",
        );
        if (confirm !== "Phục hồi") return { cancelled: true };
        const result = await this.backend.request<Record<string, unknown>>("/knowledge/rollback", {
          method: "POST", body: { version: version || null },
        });
        this.logActivity("restored", undefined, `Phục hồi ${version.slice(0, 10)}`);
        return result;
      }
      case "reload": {
        const result = await this.backend.request<Record<string, unknown>>("/knowledge/reload", { method: "POST", body: {} });
        this.logActivity("reloaded", undefined, "Đồng bộ kiến thức");
        return result;
      }
      case "importDefinitions":
        return this.importDefinitions();
      case "exportDefinitions":
        return this.exportDefinitions();
      case "setTheme": {
        const theme = payload.theme === "light" ? "light" : "dark";
        await this.context.globalState.update(THEME_KEY, theme);
        return { theme };
      }
      case "openPythonSettings":
        await vscode.commands.executeCommand("workbench.action.openSettings", "vietnameseKnowledgeAssistant.pythonPath");
        return { opened: true };
      case "openModelSettings":
        await vscode.commands.executeCommand("workbench.action.openSettings", "vietnameseKnowledgeAssistant.modelRoot");
        return { opened: true };
      case "openAdvancedSettings":
        await vscode.commands.executeCommand("workbench.action.openSettings", "vietnameseKnowledgeAssistant.");
        return { opened: true };
      case "checkBackend":
        return this.backend.start();
      case "restartBackend":
        await this.backend.stop();
        return this.backend.start();
      case "insertEditorOutput":
        return this.insertEditorOutput(String(payload.output ?? ""));
      case "dismissEditorPreview":
        this.pendingEditorOutput = undefined;
        this.pendingEditorTarget = undefined;
        return { dismissed: true };
      default:
        throw new Error("Thao tác không được hỗ trợ.");
    }
  }

  private filterDisabled(result: Record<string, any>): Record<string, any> {
    const statuses = this.context.globalState.get<Record<string, string>>(STATUS_KEY, {});
    const isPaused = (intent: unknown) => typeof intent === "string" && statuses[intent] === "paused";
    if (Array.isArray(result.candidates)) {
      result.candidates = result.candidates.filter((candidate: { intent?: string }) => !isPaused(candidate.intent));
      if (result.status === "need_choice" && result.candidates.length === 0) {
        return { ...result, status: "unknown", intent: null, message: "Không tìm thấy kiến thức đang hoạt động phù hợp." };
      }
    }
    if (isPaused(result.intent)) {
      return { ...result, status: "unknown", intent: null, latex_code: null, outputs: {}, message: "Kiến thức này đang tạm dừng." };
    }
    return result;
  }

  private activities(): Activity[] {
    return this.context.globalState.get<Activity[]>(ACTIVITY_KEY, []);
  }

  private logActivity(action: string, intent: string | undefined, title: string): void {
    const items = this.activities();
    const event: Activity = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
      action,
      intent,
      title,
      at: new Date().toISOString(),
    };
    void this.context.globalState.update(ACTIVITY_KEY, [event, ...items].slice(0, 500));
  }

  private async importDefinitions(): Promise<unknown> {
    const selected = await vscode.window.showOpenDialog({
      canSelectMany: false,
      filters: { "Tệp kiến thức JSON": ["json"] },
      openLabel: "Kiểm tra tệp",
    });
    if (!selected?.[0]) return { cancelled: true };
    const parsed = JSON.parse(await fs.readFile(selected[0].fsPath, "utf8")) as { definitions?: Definition[] } | Definition[];
    const definitions = Array.isArray(parsed) ? parsed : parsed.definitions;
    if (!Array.isArray(definitions)) throw new Error("Tệp cần là danh sách kiến thức hoặc object có trường definitions.");
    const validation = await this.backend.request<{ valid: boolean; conflicts?: string[]; errors?: unknown[] }>(
      "/knowledge/import/validate", { method: "POST", body: { definitions } },
    );
    if (!validation.valid) throw new Error(JSON.stringify(validation.errors));
    const conflicts = validation.conflicts ?? [];
    let overwrite = false;
    if (conflicts.length) {
      const choice = await vscode.window.showWarningMessage(
        `${conflicts.length} kiến thức đã tồn tại. Tệp được kiểm tra hợp lệ; có muốn ghi đè sau khi tạo backup không?`,
        { modal: true }, "Ghi đè có backup",
      );
      if (choice !== "Ghi đè có backup") return { cancelled: true, conflicts };
      overwrite = true;
    }
    const result = await this.backend.request<{ count: number }>("/knowledge/import", {
      method: "POST", body: { definitions, overwrite },
    });
    this.logActivity("imported", undefined, `Nhập ${result.count} kiến thức`);
    return result;
  }

  private async exportDefinitions(): Promise<unknown> {
    const result = await this.backend.request<{ definitions: unknown[] }>("/knowledge/export");
    const destination = await vscode.window.showSaveDialog({
      defaultUri: vscode.Uri.file(path.join(this.context.globalStorageUri.fsPath, "knowledge-export.json")),
      filters: { JSON: ["json"] },
      saveLabel: "Xuất kiến thức",
    });
    if (!destination) return { cancelled: true };
    await fs.mkdir(path.dirname(destination.fsPath), { recursive: true });
    await fs.writeFile(destination.fsPath, `${JSON.stringify({ definitions: result.definitions }, null, 2)}\n`, "utf8");
    this.logActivity("exported", undefined, `Xuất ${result.definitions.length} kiến thức`);
    return { exported: true, count: result.definitions.length };
  }

  private async runDirective(uri?: vscode.Uri, lineNumber?: number): Promise<void> {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      void vscode.window.showInformationMessage("Mở tệp và đặt con trỏ trong một chỉ thị A: trước.");
      return;
    }
    const line = Number.isInteger(lineNumber) ? lineNumber! : editor.selection.active.line;
    const document = uri && uri.toString() !== editor.document.uri.toString()
      ? await vscode.workspace.openTextDocument(uri)
      : editor.document;
    const directive = parseKnowledgeDirective(document.languageId, document.lineAt(line).text);
    if (!directive) {
      void vscode.window.showInformationMessage("Dòng hiện tại chưa có chỉ thị. Thêm comment dạng A: yêu cầu của bạn.");
      return;
    }
    const targetEditor = document.uri.toString() === editor.document.uri.toString()
      ? editor
      : await vscode.window.showTextDocument(document);
    this.pendingEditorTarget = {
      editor: targetEditor,
      range: document.lineAt(line).range,
      documentVersion: document.version,
    };
    await this.openPanel("dashboard");
    let result = await this.backend.request<Record<string, any>>("/ask", {
      method: "POST", body: { text: directive, target: "latex" },
    });
    if (result.status === "need_choice") {
      result = this.filterDisabled(result);
      const candidates = (result.candidates ?? []) as Array<{ intent: string; describe: string }>;
      const selected = await vscode.window.showQuickPick(
        candidates.map((item) => ({ label: item.describe || item.intent, description: item.intent, intent: item.intent })),
        { title: "Chọn kiến thức phù hợp", placeHolder: "Chọn một kết quả rồi xem trước" },
      );
      if (!selected) return;
      result = await this.backend.request<Record<string, any>>("/ask/choose", {
        method: "POST", body: { text: directive, candidate: selected.intent, target: "latex", values: {} },
      });
    }
    const enteredValues: Record<string, string> = {};
    while (result.status === "need_slots" && Array.isArray(result.missing_slots)) {
      const field = String(result.missing_slots[0] ?? "").replace(/^\$\./, "");
      const value = await vscode.window.showInputBox({
        title: "Bổ sung thông tin",
        prompt: `Nhập ${field}`,
        placeHolder: "Giá trị sẽ được kiểm tra trước khi tạo kết quả",
        ignoreFocusOut: true,
      });
      if (value === undefined) return;
      enteredValues[field] = value;
      result = await this.backend.request<Record<string, any>>("/ask/choose", {
        method: "POST", body: { text: directive, candidate: String(result.intent ?? ""), target: "latex", values: enteredValues },
      });
    }
    if (result.status !== "ok" || typeof result.latex_code !== "string") {
      const message = result.message || "Chưa tìm thấy kiến thức phù hợp cho chỉ thị này.";
      const action = await vscode.window.showWarningMessage(message, "Tạo kiến thức mới");
      if (action === "Tạo kiến thức mới") this.openPanel("builder");
      return;
    }
    this.pendingEditorOutput = result.latex_code;
    this.panel?.webview.postMessage({ type: "editorPreview", output: result.latex_code, directive });
  }

  private async insertEditorOutput(output: string): Promise<{ inserted: boolean }> {
    const target = this.pendingEditorTarget;
    if (!target) return { inserted: false };
    if (target.editor.document.version !== target.documentVersion) {
      this.pendingEditorTarget = undefined;
      this.pendingEditorOutput = undefined;
      throw new Error("Tệp đã thay đổi sau khi mở preview. Chạy lại chỉ thị để tránh ghi đè nội dung mới.");
    }
    const inserted = await target.editor.edit((edit) => edit.replace(target.range, output));
    if (!inserted) throw new Error("VS Code không thể chèn kết quả vào vùng đã chọn.");
    this.pendingEditorTarget = undefined;
    this.pendingEditorOutput = undefined;
    this.logActivity("inserted", undefined, "Chèn kết quả vào Editor");
    return { inserted: true };
  }
}

class DirectiveCodeActionProvider implements vscode.CodeActionProvider {
  provideCodeActions(document: vscode.TextDocument, range: vscode.Range): vscode.CodeAction[] {
    const line = document.lineAt(range.start.line);
    const directive = parseKnowledgeDirective(document.languageId, line.text);
    if (!directive) return [];
    const action = new vscode.CodeAction(
      "Xem trước kiến thức: " + directive,
      vscode.CodeActionKind.QuickFix,
    );
    action.command = {
      command: "vietnameseKnowledgeAssistant.runDirective",
      title: "Tạo preview",
      arguments: [document.uri, line.lineNumber],
    };
    return [action];
  }
}

function randomNonce(): string {
  return randomBytes(24).toString("base64");
}

let activeExtension: KnowledgeExtension | undefined;

export function activate(context: vscode.ExtensionContext): void {
  activeExtension = new KnowledgeExtension(context);
  activeExtension.activate();
}

export async function deactivate(): Promise<void> {
  await activeExtension?.deactivate();
  activeExtension = undefined;
}
