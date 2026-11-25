export type BridgeResponse<T> = T;

type VsCodeApi = { postMessage(message: unknown): void; getState(): unknown; setState(state: unknown): void };
declare global { interface Window { acquireVsCodeApi?: () => VsCodeApi } }

const vscode = window.acquireVsCodeApi?.();
const pending = new Map<string, { resolve: (value: unknown) => void; reject: (reason: Error) => void }>();
let sequence = 0;

window.addEventListener("message", (event: MessageEvent) => {
  const message = event.data as { type?: string; id?: string; data?: unknown; error?: string };
  if (message.type !== "response" || !message.id) return;
  const entry = pending.get(message.id);
  if (!entry) return;
  pending.delete(message.id);
  if (message.error) entry.reject(new Error(message.error));
  else entry.resolve(message.data);
});

export function request<T>(action: string, payload: Record<string, unknown> = {}): Promise<T> {
  if (!vscode) return Promise.reject(new Error("Extension Host chưa sẵn sàng."));
  const id = `${Date.now()}-${sequence++}`;
  return new Promise<T>((resolve, reject) => {
    pending.set(id, { resolve: resolve as (value: unknown) => void, reject });
    vscode.postMessage({ id, action, payload });
  });
}

export function subscribe(listener: (message: Record<string, unknown>) => void): () => void {
  const handler = (event: MessageEvent) => {
    if (event.source === window && event.data && typeof event.data === "object") listener(event.data as Record<string, unknown>);
  };
  window.addEventListener("message", handler);
  vscode?.postMessage({ type: "ready" });
  return () => window.removeEventListener("message", handler);
}
