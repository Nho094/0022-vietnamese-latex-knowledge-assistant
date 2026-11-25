import React, { type ErrorInfo, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";

class StartupErrorBoundary extends React.Component<{ children: ReactNode }, { error: string }> {
  state = { error: "" };

  static getDerivedStateFromError(error: unknown) {
    return { error: error instanceof Error ? error.message : String(error) };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Knowledge Assistant Webview render failed", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return <main className="startup-error">
        <h2>Giao diện gặp lỗi khi khởi chạy</h2>
        <pre>{this.state.error}</pre>
        <p>Mở View → Output → Vietnamese Knowledge Assistant để kiểm tra backend.</p>
      </main>;
    }
    return this.props.children;
  }
}

const root = document.getElementById("root");
if (!root) throw new Error("Webview root element is missing.");
const bootStatus = document.getElementById("boot-status");
const observer = new MutationObserver(() => {
  if (root.childElementCount > 0) {
    bootStatus?.remove();
    observer.disconnect();
  }
});
observer.observe(root, { childList: true });
createRoot(root).render(<React.StrictMode><StartupErrorBoundary><App /></StartupErrorBoundary></React.StrictMode>);
