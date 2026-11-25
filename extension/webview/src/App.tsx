import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import {
  Activity as ActivityIcon, ArrowDownToLine, ArrowLeft, ArrowRight, BookOpen, Check, ChevronLeft, ChevronRight,
  CircleAlert, Clock3, Code2, Database, FilePlus2, History, Home, LoaderCircle,
  Moon, MoreHorizontal, Search, Settings2, Sparkles, Sun, Trash2, Upload, WandSparkles,
} from "lucide-react";
import { request, subscribe } from "./bridge";
import { Button } from "./components/ui/Button";
import { Card, CardHeader, CardTitle } from "./components/ui/Card";
import { FormField, Input, Select, Textarea } from "./components/ui/Field";
import { Table, TableCell, TableHead } from "./components/ui/Table";
import { Builder, categories, definitionToDraft, universalTargets } from "./pages/Builder";
import type { Activity, CategoryOption, Definition, MilestoneStatus, ServiceStatus } from "./types";

type Page = "dashboard" | "knowledge" | "builder" | "detail" | "history" | "settings";
type Candidate = { intent: string; describe?: string; score?: number; category?: string };
type AskResult = {
  status: string;
  intent?: string | null;
  target?: string | null;
  message?: string;
  latex_code?: string | null;
  outputs?: Record<string, string>;
  requires?: string[];
  missing_slots?: string[];
  slots?: Record<string, unknown>;
  slot_errors?: string[];
  candidates?: Candidate[];
};
type Bootstrap = {
  definitions: Definition[];
  categories: string[];
  universal_categories?: CategoryOption[];
  corpus?: { corpora?: Array<Record<string, unknown>>; items?: Array<Record<string, unknown>> };
  history?: { active_version?: string; versions?: Array<{ version: string; created_at: string; active: boolean; valid_hash: boolean }> };
  system?: {
    status?: string;
    message?: string;
    python?: string;
    pipeline_mode?: string;
    classifier_available?: boolean;
    classifier_message?: string;
    classifier_checkpoint_path?: string;
    phobert_base_path?: string;
    classifier_missing_dependencies?: string[];
    retrieval_model_available?: boolean;
    retrieval_model_message?: string;
    retrieval_model_path?: string;
    retrieval_index_available?: boolean;
    retrieval_index_message?: string;
    retrieval_index_path?: string;
    retrieval_missing_dependencies?: string[];
    model_root?: string;
    retrieval_cache_available?: boolean;
    milestone?: MilestoneStatus;
  };
  builder?: { suggestions_available?: boolean; manual_builder_available?: boolean; message?: string };
  activity: Activity[];
  theme: string;
  initialPage?: Page;
  editorPreview?: string;
  service?: ServiceStatus;
};

const nav = [
  { id: "dashboard" as const, label: "Tổng quan", icon: Home },
  { id: "knowledge" as const, label: "Kiến thức của tôi", icon: BookOpen },
  { id: "builder" as const, label: "Thêm kiến thức", icon: FilePlus2 },
  { id: "history" as const, label: "Lịch sử", icon: History },
  { id: "settings" as const, label: "Cài đặt", icon: Settings2 },
];

const actionLabels: Record<string, string> = {
  created: "Đã tạo kiến thức", imported: "Đã nhập kiến thức", exported: "Đã xuất kiến thức",
  updated: "Đã cập nhật kiến thức",
  paused: "Đã tạm dừng", activated: "Đã kích hoạt", restored: "Đã phục hồi",
  deleted: "Đã xóa kiến thức", inserted: "Đã chèn vào Editor", reloaded: "Đã đồng bộ kiến thức",
  generated: "Đã xuất ra Output Panel", trained: "Đã tối ưu Retrieval theo mốc",
};

function displayName(item: Definition): string {
  return item.display_name || item.intent.replaceAll("_", " ").toLocaleLowerCase("vi");
}

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : new Intl.DateTimeFormat("vi-VN", { dateStyle: "medium", timeStyle: "short" }).format(date);
}

export default function App() {
  const [data, setData] = useState<Bootstrap | null>(null);
  const [page, setPage] = useState<Page>("dashboard");
  const [selected, setSelected] = useState<Definition | undefined>();
  const [editDefinition, setEditDefinition] = useState<Definition | undefined>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [toast, setToast] = useState("");
  const [service, setService] = useState<ServiceStatus | undefined>();
  const [editorPreview, setEditorPreview] = useState("");

  const refresh = useCallback(async (showLoading = true) => {
    if (showLoading) setLoading(true);
    setError("");
    try {
      const result = await request<Bootstrap>("bootstrap");
      setData(result);
      setPage((current) => current === "dashboard" ? result.initialPage ?? current : current);
      setService(result.service);
      setEditorPreview(result.editorPreview ?? "");
      document.documentElement.dataset.theme = result.theme === "light" ? "light" : "dark";
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Không tải được dữ liệu.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const unsubscribe = subscribe((message) => {
      if (message.type === "serviceStatus") setService(message.status as ServiceStatus);
      if (message.type === "navigate" && typeof message.page === "string") setPage(message.page as Page);
      if (message.type === "editorPreview" && typeof message.output === "string") setEditorPreview(message.output);
    });
    void refresh();
    return unsubscribe;
  }, [refresh]);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(""), 3200);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const navigate = (next: Page) => {
    setPage(next);
    setEditDefinition(undefined);
    if (next !== "detail") setSelected(undefined);
  };

  const startEdit = async (item: Definition) => {
    try {
      const full = await request<Definition>("getDefinition", { intent: item.intent });
      definitionToDraft(full);
      setEditDefinition(full);
      setPage("builder");
    } catch (reason) {
      setToast(reason instanceof Error ? reason.message : "Không mở được biểu mẫu chỉnh sửa.");
    }
  };

  const handlePublished = async (intent: string) => {
    setEditDefinition(undefined);
    await refresh(false);
    const match = data?.definitions.find((item) => item.intent === intent);
    setToast(match ? `Đã cập nhật “${displayName(match)}”.` : "Đã lưu kiến thức. Bộ tìm kiếm đang đồng bộ.");
    setPage("knowledge");
  };

  const toggleTheme = async () => {
    const theme = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = theme;
    try {
      await request("setTheme", { theme });
      setData((current) => current ? { ...current, theme } : current);
    } catch (reason) {
      setToast(reason instanceof Error ? reason.message : "Không lưu được giao diện.");
    }
  };

  const serviceState = service?.state ?? "starting";
  const serviceLabel = serviceState === "ready" ? "Đang hoạt động" : serviceState === "degraded" ? "Chế độ hạn chế" : serviceState === "starting" ? "Đang khởi động" : serviceState === "missing" ? "Thiếu Python" : serviceState === "error" ? "Lỗi dịch vụ / thư viện" : "Đang kiểm tra";

  return <div className="app-shell">
    <aside className="sidebar">
      <button className="brand" onClick={() => navigate("dashboard")} aria-label="Về tổng quan">
        <span className="brand-mark"><WandSparkles size={17} /></span><span className="brand-text"><strong>Trợ lý kiến thức</strong><small>Vietnamese Knowledge Assistant</small></span>
      </button>
      <div className="sidebar-section-label">KHÔNG GIAN CỦA BẠN</div>
      <nav className="sidebar-nav" aria-label="Điều hướng chính">
        {nav.slice(0, 3).map(({ id, label, icon: Icon }) => <button key={id} className={`nav-item ${page === id || (id === "knowledge" && page === "detail") ? "active" : ""}`} onClick={() => navigate(id)} aria-current={page === id ? "page" : undefined}><Icon size={16} /><span>{label}</span>{id === "knowledge" && data && <span className="nav-count">{data.definitions?.length ?? 0}</span>}</button>)}
        <div className="sidebar-section-label sidebar-section-spaced">HỆ THỐNG</div>
        {nav.slice(3).map(({ id, label, icon: Icon }) => <button key={id} className={`nav-item ${page === id ? "active" : ""}`} onClick={() => navigate(id)} aria-current={page === id ? "page" : undefined}><Icon size={16} /><span>{label}</span></button>)}
      </nav>
      <div className="sidebar-bottom">
        <div className="runtime-label">TRẠNG THÁI CỤC BỘ</div>
        <RuntimeRow label="Backend" value={serviceLabel} state={serviceState} />
        <RuntimeRow label="PhoBERT" value={service?.classifierAvailable ? "Sẵn sàng" : "Chưa khả dụng"} state={service?.classifierAvailable ? "ready" : "missing"} />
        <RuntimeRow label="Retrieval model" value={service?.retrievalModelAvailable ? "Sẵn sàng" : "Chưa khả dụng"} state={service?.retrievalModelAvailable ? "ready" : "missing"} />
        <RuntimeRow label="Retrieval index" value={service?.retrievalIndexAvailable ? "Đã tạo" : "Sẽ tạo lần đầu"} state={service?.retrievalIndexAvailable ? "ready" : "starting"} />
        <button className="sidebar-theme" onClick={() => void toggleTheme()}>{document.documentElement.dataset.theme === "light" ? <Moon size={15} /> : <Sun size={15} />}<span>Đổi giao diện</span></button>
      </div>
    </aside>

    <main className="main-area">
      <header className="topbar">
        <div className="breadcrumbs"><span>Không gian</span><ChevronRight size={13} /><strong>{nav.find((item) => item.id === page)?.label ?? (page === "detail" ? "Chi tiết kiến thức" : "Tạo kiến thức")}</strong></div>
        <div className="topbar-actions"><span className={`small-status ${serviceState}`}><span className="service-dot" />{serviceLabel}</span><Button variant="secondary" size="icon" aria-label="Đổi giao diện" onClick={() => void toggleTheme()}>{document.documentElement.dataset.theme === "light" ? <Moon size={15} /> : <Sun size={15} />}</Button><Button size="sm" onClick={() => navigate("builder")}><FilePlus2 size={14} /> Thêm kiến thức</Button></div>
      </header>

      <div className="content-area">
        {loading && !data && <div className="full-state"><LoaderCircle className="spin" size={24} /><strong>Đang kết nối dịch vụ</strong><span>Đọc danh sách kiến thức và trạng thái cục bộ…</span></div>}
        {error && !data && <div className="full-state error-state"><CircleAlert size={24} /><strong>Chưa tải được dữ liệu</strong><span>{error}</span><Button variant="secondary" onClick={() => void refresh()}>Thử lại</Button><Button variant="ghost" onClick={() => void request("openPythonSettings")}>Mở cài đặt Python</Button></div>}
        {data && page === "dashboard" && <Dashboard data={data} service={service} onNavigate={navigate} onCreate={() => navigate("builder")} onToast={setToast} onInsert={async (output) => {
          try {
            const result = await request<{ inserted: boolean }>("insertEditorOutput", { output });
            if (result.inserted) { setEditorPreview(""); setToast("Đã chèn kết quả vào Editor."); }
          } catch (reason) { setToast(reason instanceof Error ? reason.message : "Không chèn được."); }
        }} editorPreview={editorPreview} onDismissEditor={() => { setEditorPreview(""); void request("dismissEditorPreview"); }} />}
        {data && page === "knowledge" && <KnowledgeExplorer data={data} onOpen={(item) => { setSelected(item); setPage("detail"); }} onCreate={() => navigate("builder")} onEdit={startEdit} onChanged={() => void refresh(false)} onToast={setToast} />}
        {data && page === "detail" && selected && <KnowledgeDetail item={selected} onBack={() => navigate("knowledge")} onEdit={() => void startEdit(selected)} onChanged={() => void refresh(false)} onToast={setToast} />}
        {data && page === "builder" && <Builder initialDefinition={editDefinition} categoryOptions={data.universal_categories} onBack={() => navigate("knowledge")} onPublished={(intent) => void handlePublished(intent)} />}
        {data && page === "history" && <HistoryPage data={data} onChanged={() => void refresh(false)} onToast={setToast} />}
        {data && page === "settings" && <SettingsPage data={data} service={service} onRefresh={() => void refresh()} onToast={setToast} />}
      </div>
    </main>
    {toast && <div className="toast" role="status"><Check size={15} />{toast}</div>}
  </div>;
}

function Dashboard({ data, service, onNavigate, onCreate, onToast, onInsert, editorPreview, onDismissEditor }: {
  data: Bootstrap;
  service?: ServiceStatus;
  onNavigate: (page: Page) => void;
  onCreate: () => void;
  onToast: (message: string) => void;
  onInsert: (output: string) => void;
  editorPreview: string;
  onDismissEditor: () => void;
}) {
  const active = data.definitions.filter((item) => item.assistant_status !== "paused").length;
  const review = data.definitions.filter((item) => item.review_needed === true).length;
  const recent = (data.activity ?? []).slice(0, 4);
  const recentIntents = new Set<string>();
  const recentlyChanged = (data.activity ?? []).flatMap((event) => {
    if (!event.intent || recentIntents.has(event.intent)) return [];
    const definition = data.definitions.find((item) => item.intent === event.intent);
    if (!definition) return [];
    recentIntents.add(event.intent);
    return [{ definition, event }];
  }).slice(0, 4);
  const liveService = service ?? data.service;
  const classifierReady = liveService?.classifierAvailable ?? data.system?.classifier_available ?? false;
  const retrievalReady = liveService?.retrievalModelAvailable ?? data.system?.retrieval_model_available ?? false;
  const liveState = liveService?.state ?? "starting";
  const showRuntimeNotice = liveState === "degraded" || liveState === "error" || (liveState === "starting" && (data.system?.status === "degraded" || data.system?.status === "error"));
  const runtimeNoticeMessage = liveService?.message || data.system?.message || "Dịch vụ đang chạy ở chế độ hạn chế.";
  return <div className="page dashboard-page">
    <div className="dashboard-hero"><div><div className="eyebrow">KHÔNG GIAN KIẾN THỨC CỤC BỘ</div><h1>Chào bạn</h1><p>Tạo, kiểm tra và dùng kiến thức ngay trong VS Code.</p></div><Button onClick={() => onNavigate("builder")}><FilePlus2 size={15} /> Tạo kiến thức</Button></div>
    {showRuntimeNotice && <div className="notice warning"><CircleAlert size={17} /><span>{runtimeNoticeMessage}</span><button onClick={() => onNavigate("settings")}>Xem trạng thái</button></div>}
    <div className="runtime-overview" aria-label="Trạng thái mô hình">
      <RuntimeSummary label="Backend" value={liveState === "ready" ? "Sẵn sàng" : liveState === "degraded" ? "Chế độ hạn chế" : liveState === "error" ? "Lỗi thư viện / dịch vụ" : liveState === "starting" ? "Đang khởi động" : "Cần kiểm tra"} state={liveState} />
      <RuntimeSummary label="PhoBERT" value={classifierReady ? "Checkpoint đã có" : "Chưa có checkpoint"} state={classifierReady ? "ready" : "missing"} />
      <RuntimeSummary label="Retrieval model" value={retrievalReady ? "Model đã có" : "Chưa có model"} state={retrievalReady ? "ready" : "missing"} />
    </div>
    <div className="stats-grid">
      <StatCard label="Tổng kiến thức" value={data.definitions.length} detail="Được quản lý trong registry" icon={<BookOpen size={17} />} />
      <StatCard label="Đang hoạt động" value={active} detail="Có thể dùng trong yêu cầu mới" icon={<Check size={17} />} tone="green" />
      <StatCard label="Cần xem lại" value={review} detail="Đang chờ kiểm tra nội dung" icon={<CircleAlert size={17} />} tone="amber" />
    </div>
    <Card className="workspace-card"><CardHeader><div><div className="workspace-kicker"><Sparkles size={13} /> AI WORKSPACE</div><CardTitle>Hỏi bằng ngôn ngữ tự nhiên</CardTitle><p className="workspace-description">Nhập yêu cầu, chọn định dạng (hoặc để Tự động) và kiểm tra kết quả trước khi dùng.</p></div><span className="workspace-local-tag"><span className={`service-dot ${liveState}`} />Xử lý cục bộ</span></CardHeader><QuickAsk definitions={data.definitions} onCreate={onCreate} onToast={onToast} /></Card>
    <div className="dashboard-grid">
      <Card><CardHeader><div><div className="eyebrow">HOẠT ĐỘNG</div><CardTitle>Gần đây</CardTitle></div><Button variant="ghost" size="sm" onClick={() => onNavigate("history")}>Xem lịch sử <ArrowRight size={13} /></Button></CardHeader>
        {recent.length ? <div className="activity-list">{recent.map((event) => <ActivityRow key={event.id} item={event} />)}</div> : <div className="empty-state"><ActivityIcon size={19} /><strong>Chưa có hoạt động</strong><span>Các thao tác của bạn sẽ được ghi lại tại đây.</span></div>}
      </Card>
      <Card><CardHeader><div><div className="eyebrow">THAY ĐỔI GẦN ĐÂY</div><CardTitle>Kiến thức vừa cập nhật</CardTitle></div><Button variant="ghost" size="sm" onClick={() => onNavigate("knowledge")}>Mở thư viện <ArrowRight size={13} /></Button></CardHeader>
        {recentlyChanged.length ? <div className="recent-grid">{recentlyChanged.map(({ definition, event }) => <button className="recent-knowledge" key={`${event.id}-${definition.intent}`} onClick={() => onNavigate("knowledge")}><span className="knowledge-icon"><Code2 size={16} /></span><span><strong>{displayName(definition)}</strong><small>{formatDate(event.at)} · {definition.describe || "Chưa có mô tả"}</small></span><ChevronRight size={15} /></button>)}</div> : <div className="empty-state compact"><Clock3 size={17} /><span>Chưa có kiến thức nào được chỉnh sửa trong Extension.</span></div>}
      </Card>
    </div>
    {editorPreview && <EditorPreview output={editorPreview} onInsert={onInsert} onDismiss={onDismissEditor} />}
  </div>;
}

function StatCard({ label, value, detail, icon, tone = "purple" }: { label: string; value: number; detail: string; icon: ReactNode; tone?: string }) {
  return <Card className="stat-card"><span className={`stat-icon ${tone}`}>{icon}</span><div className="stat-copy"><span>{label}</span><strong>{value.toLocaleString("vi-VN")}</strong><small>{detail}</small></div></Card>;
}

function RuntimeRow({ label, value, state }: { label: string; value: string; state: string }) {
  return <div className="runtime-row"><span className={`runtime-dot ${state}`} /><span className="runtime-name">{label}</span><span className="runtime-value">{value}</span></div>;
}

function RuntimeSummary({ label, value, state }: { label: string; value: string; state: string }) {
  return <div className="runtime-summary"><span className={`runtime-dot ${state}`} /><span className="runtime-summary-copy"><small>{label}</small><strong>{value}</strong></span></div>;
}

function QuickAsk({ definitions, onCreate, onToast }: { definitions: Definition[]; onCreate: () => void; onToast?: (message: string) => void }) {
  const [text, setText] = useState("");
  const [target, setTarget] = useState("auto");
  const [result, setResult] = useState<AskResult | null>(null);
  const [values, setValues] = useState<Record<string, unknown>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [inserted, setInserted] = useState(false);
  const activeIntent = result?.intent ? definitions.find((item) => item.intent === result.intent) : undefined;
  const missing = (result?.missing_slots ?? []).map((name) => name.replace(/^\$\./, ""));
  const fields = activeIntent?.input_schema?.properties ?? {};
  const extractedSlots = Object.entries(result?.slots ?? {});
  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!text.trim()) return;
    setBusy(true); setError(""); setInserted(false); setValues({});
    try {
      const response = await request<AskResult>("ask", { text, target });
      setResult(response);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Không xử lý được yêu cầu."); }
    finally { setBusy(false); }
  };
  const choose = async (candidate: Candidate) => {
    setBusy(true); setError("");
    try {
      setResult(await request<AskResult>("choose", { text, target, candidate: candidate.intent, values: {}, record_feedback: true }));
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Không chọn được kiến thức."); }
    finally { setBusy(false); }
  };
  const continueWithValues = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!result?.intent) return;
    setBusy(true); setError("");
    try { setResult(await request<AskResult>("choose", { text, target, candidate: result.intent, values, record_feedback: false })); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Không tiếp tục được yêu cầu."); }
    finally { setBusy(false); }
  };
  const outputKeys = result?.outputs ? Object.keys(result.outputs) : [];
  const resolvedTarget = (target !== "auto" && result?.outputs?.[target] ? target : undefined)
    ?? result?.target
    ?? outputKeys[0]
    ?? "";
  const firstOutput = result?.outputs
    ? Object.values(result.outputs).find((value) => typeof value === "string" && value.length > 0)
    : undefined;
  const output = (target !== "auto" ? result?.outputs?.[target] : undefined)
    ?? (resolvedTarget ? result?.outputs?.[resolvedTarget] : undefined)
    ?? firstOutput
    ?? result?.latex_code
    ?? undefined;
  const effectiveTarget = resolvedTarget || (target !== "auto" ? target : "");
  const requiredPackages = Array.isArray(result?.requires) && result.requires.length > 0
    ? result.requires
    : Array.isArray(activeIntent?.requires)
      ? (activeIntent.requires as string[])
      : [];
  const rememberAlias = async () => {
    if (!result?.intent || !text.trim()) return;
    try {
      await request("learnAlias", { intent: result.intent, query: text.trim() });
      onToast?.("Đã ghi nhớ cách hỏi này vào Retrieval.");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Không lưu được alias.");
    }
  };
  return <div className="quick-ask">
    <form onSubmit={(event) => void submit(event)} className="quick-ask-form">
      <label className="ask-input-label" htmlFor="quick-ask-input">YÊU CẦU CỦA BẠN</label>
      <Textarea id="quick-ask-input" rows={4} value={text} onChange={(event) => { setText(event.target.value); setResult(null); }} placeholder="Ví dụ: Setup Sunshine trên Linux, Viết hàm Fibonacci Python, hoặc Tạo phân số 2/5" aria-label="Mô tả điều bạn cần" />
      <div className="quick-ask-footer"><div className="target-picker"><span>Định dạng</span><Select aria-label="Kiểu kết quả" value={target} onChange={(event) => { setTarget(event.target.value); setResult(null); setInserted(false); }}><option value="auto">Tự động (theo kiến thức)</option>{universalTargets.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</Select></div><Button type="submit" size="md" disabled={busy || !text.trim()}>{busy ? <><LoaderCircle className="spin" size={14} /> Đang xử lý</> : <><Sparkles size={14} /> Tạo kết quả</>}</Button></div>
    </form>
    <div className="quick-ask-response" aria-live="polite">
      {error && <div className="inline-error" role="alert">{error}</div>}
      {busy && <div className="response-placeholder"><LoaderCircle className="spin" size={18} /><strong>Đang xử lý yêu cầu</strong><span>Đang kiểm tra kiến thức phù hợp…</span></div>}
      {!busy && !error && !result && <div className="response-placeholder"><span className="response-symbol"><Sparkles size={18} /></span><strong>Kết quả sẽ xuất hiện ở đây</strong><span>Trợ lý dùng dữ liệu và mô hình có sẵn trên máy này.</span></div>}
      {!busy && result?.status === "need_choice" && <div className="choice-list"><strong>Chọn kiến thức phù hợp</strong>{(result.candidates ?? []).map((candidate) => <button key={candidate.intent} onClick={() => void choose(candidate)} disabled={busy}><span><b>{candidate.describe || displayName({ intent: candidate.intent })}</b><small>{candidate.intent}</small></span><ChevronRight size={14} /></button>)}<Button size="sm" variant="ghost" onClick={onCreate}>Không đúng? Thêm kiến thức mới</Button></div>}
      {!busy && result?.status === "need_slots" && missing.length > 0 && <form className="missing-fields" onSubmit={(event) => void continueWithValues(event)}><strong>Bổ sung thông tin còn thiếu</strong>{extractedSlots.length > 0 && <span className="slot-summary">Đã nhận: {extractedSlots.map(([key, value]) => `${key} = ${String(value)}`).join(" · ")}. Hãy nhập các giá trị còn thiếu bên dưới.</span>}{missing.map((key) => { const field = fields[key]; const numeric = field?.type === "number" || field?.type === "integer"; return <FormField key={key} label={field?.title ? `${field.title} (${key})` : field?.description ? `${key} — ${field.description}` : key}><Input type={numeric ? "number" : "text"} step={numeric ? "any" : undefined} value={String(values[key] ?? "")} onChange={(event) => setValues((current) => ({ ...current, [key]: event.target.value }))} placeholder={`Nhập ${field?.title ?? key}`} /></FormField>; })}<Button type="submit" size="sm" disabled={busy}>Tiếp tục</Button></form>}
      {!busy && result?.status === "need_slots" && missing.length === 0 && <div className="empty-result"><CircleAlert size={14} /><span>{result.message || "Chưa xác định được trường cần bổ sung."} {result.slot_errors?.join(" ")}</span></div>}
      {!busy && result?.status === "ok" && typeof output === "string" && <div className="ask-result"><div className="result-heading"><span><Check size={14} /> Kết quả ({targetLabel(effectiveTarget || "text")})</span><div className="row-actions"><button type="button" onClick={() => void rememberAlias()}>Lưu câu hỏi</button><button type="button" onClick={() => { void navigator.clipboard?.writeText(output); setInserted(false); }}>Sao chép</button></div></div>{result.intent && <div className="ask-context"><small>KIẾN THỨC ĐƯỢC CHỌN</small><strong>{activeIntent ? displayName(activeIntent) : result.intent}</strong>{result.slots && Object.keys(result.slots).length > 0 && <span>{Object.entries(result.slots).map(([key, value]) => `${key}: ${String(value)}`).join(" · ")}</span>}</div>}<pre>{output}</pre>{effectiveTarget === "latex" && requiredPackages.length > 0 && <small className="field-hint">Gói LaTeX cần khai báo: {requiredPackages.map((pkg) => `\\usepackage{${pkg}}`).join(", ")}</small>}<small>Khi chạy chỉ thị `# A:` trong Editor, kết quả sẽ in thẳng ra bảng Output của VS Code.</small></div>}
      {!busy && result && !["ok", "need_choice", "need_slots"].includes(result.status) && <div className="empty-result"><CircleAlert size={14} />{result.message || "Chưa tìm được kiến thức phù hợp."}<Button size="sm" variant="ghost" onClick={onCreate}>Tạo kiến thức</Button></div>}
      {inserted && <span className="field-hint">Đã chèn.</span>}
    </div>
  </div>;
}

function EditorPreview({ output, onInsert, onDismiss }: { output: string; onInsert: (value: string) => void; onDismiss: () => void }) {
  return <Card className="editor-preview-card"><CardHeader><div><div className="eyebrow">PREVIEW TỪ EDITOR</div><CardTitle>Xem lại trước khi chèn</CardTitle></div><Button variant="ghost" size="icon" aria-label="Đóng preview" onClick={onDismiss}><Trash2 size={15} /></Button></CardHeader><pre className="output-preview">{output}</pre><div className="editor-preview-actions"><span>Chỉ thị trong Editor sẽ được thay bằng kết quả sau khi xác nhận.</span><Button onClick={() => onInsert(output)}><Check size={14} /> Chèn vào Editor</Button></div></Card>;
}

function KnowledgeExplorer({ data, onOpen, onCreate, onEdit, onChanged, onToast }: {
  data: Bootstrap;
  onOpen: (item: Definition) => void;
  onCreate: () => void;
  onEdit: (item: Definition) => void;
  onChanged: () => void;
  onToast: (message: string) => void;
}) {
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState("");
  const [intent, setIntent] = useState("");
  const [status, setStatus] = useState("");
  const [pageIndex, setPageIndex] = useState(0);
  const [busy, setBusy] = useState("");
  const rows = useMemo(() => data.definitions.filter((item) => {
    const haystack = `${displayName(item)} ${item.intent} ${item.describe ?? ""} ${(item.aliases ?? []).join(" ")}`.toLocaleLowerCase("vi");
    return (!search || haystack.includes(search.toLocaleLowerCase("vi")))
      && (!category || item.category === category)
      && (!intent || item.intent === intent)
      && (!status || (status === "active" ? item.assistant_status !== "paused" : item.assistant_status === "paused"));
  }), [data.definitions, search, category, intent, status]);
  const pageSize = 12;
  const pageCount = Math.max(1, Math.ceil(rows.length / pageSize));
  const visibleRows = rows.slice(pageIndex * pageSize, (pageIndex + 1) * pageSize);
  useEffect(() => setPageIndex((current) => Math.min(current, pageCount - 1)), [pageCount]);
  const toggle = async (item: Definition) => {
    setBusy(item.intent);
    try {
      await request("toggleKnowledge", { intent: item.intent, title: displayName(item), enabled: item.assistant_status === "paused" });
      onToast(item.assistant_status === "paused" ? "Đã kích hoạt kiến thức." : "Đã tạm dừng kiến thức trong Extension.");
      onChanged();
    } catch (reason) { onToast(reason instanceof Error ? reason.message : "Không cập nhật được trạng thái."); }
    finally { setBusy(""); }
  };
  const remove = async (item: Definition) => {
    setBusy(item.intent);
    try {
      const result = await request<{ deleted?: boolean; cancelled?: boolean }>("deleteDefinition", { intent: item.intent, title: displayName(item) });
      if (result.deleted) { onToast("Đã xóa kiến thức. Phiên bản trước có thể phục hồi từ Lịch sử."); onChanged(); }
    } catch (reason) { onToast(reason instanceof Error ? reason.message : "Không xóa được kiến thức."); }
    finally { setBusy(""); }
  };
  return <div className="page explorer-page"><div className="page-heading"><div><div className="eyebrow">THƯ VIỆN RIÊNG</div><h1>Kiến thức của tôi</h1><p>Tìm kiếm, kiểm tra và quản lý những kiến thức trợ lý có thể dùng.</p></div><Button onClick={onCreate}><FilePlus2 size={14} /> Thêm kiến thức</Button></div>
    <Card className="explorer-card"><div className="explorer-toolbar"><label className="search-box"><Search size={15} /><input value={search} onChange={(event) => { setSearch(event.target.value); setPageIndex(0); }} placeholder="Tìm theo tên, mô tả hoặc câu ví dụ" aria-label="Tìm kiếm kiến thức" /></label>
      <Select value={category} onChange={(event) => { setCategory(event.target.value); setPageIndex(0); }} aria-label="Lọc nhóm"><option value="">Mọi nhóm</option>{data.categories?.map((item) => <option key={item} value={item}>{categoryLabel(item)}</option>)}</Select>
      <Select value={intent} onChange={(event) => { setIntent(event.target.value); setPageIndex(0); }} aria-label="Lọc theo Intent"><option value="">Mọi kiến thức</option>{data.definitions.map((item) => <option key={item.intent} value={item.intent}>{displayName(item)} · {item.intent}</option>)}</Select>
      <Select value={status} onChange={(event) => { setStatus(event.target.value); setPageIndex(0); }} aria-label="Lọc trạng thái"><option value="">Mọi trạng thái</option><option value="active">Đang hoạt động</option><option value="paused">Đã tạm dừng</option></Select>
      <span className="result-count">{rows.length} kiến thức</span></div>
      {rows.length ? <Table><thead><tr><TableHead>Kiến thức</TableHead><TableHead>Intent</TableHead><TableHead>Nhóm</TableHead><TableHead>Trạng thái</TableHead><TableHead>Đầu ra</TableHead><TableHead><span className="sr-only">Thao tác</span></TableHead></tr></thead><tbody>{visibleRows.map((item) => <tr key={item.intent}>
        <TableCell><button className="table-primary" onClick={() => onOpen(item)}>{displayName(item)}</button><span className="table-secondary">{item.describe || "Chưa có mô tả"}</span></TableCell>
        <TableCell><code className="intent-code">{item.intent}</code></TableCell>
        <TableCell><span className="category-pill">{categoryLabel(item.category)}</span></TableCell>
        <TableCell><span className={`status-pill ${item.assistant_status === "paused" ? "paused" : "active"}`}><span />{item.assistant_status === "paused" ? "Tạm dừng" : "Đang dùng"}</span></TableCell>
        <TableCell>{Object.keys(item.render?.templates ?? {}).map(targetLabel).join(", ") || "—"}</TableCell>
        <TableCell><div className="row-actions"><Button variant="ghost" size="sm" onClick={() => onOpen(item)}>Chi tiết</Button><Button variant="ghost" size="icon" disabled={busy === item.intent} aria-label="Thao tác khác" onClick={() => void onEdit(item)}><MoreHorizontal size={15} /></Button><Button variant="ghost" size="icon" disabled={busy === item.intent} aria-label={item.assistant_status === "paused" ? "Kích hoạt" : "Tạm dừng"} onClick={() => void toggle(item)}>{item.assistant_status === "paused" ? <Check size={15} /> : <Moon size={15} />}</Button><Button variant="ghost" size="icon" disabled={busy === item.intent} aria-label="Xóa kiến thức" onClick={() => void remove(item)}><Trash2 size={14} /></Button></div></TableCell>
      </tr>)}</tbody></Table> : <div className="empty-state"><Search size={20} /><strong>Không tìm thấy kiến thức</strong><span>Thử thay đổi từ khóa hoặc bộ lọc.</span></div>}
      <div className="table-footer"><span>{rows.length ? `${pageIndex * pageSize + 1}–${Math.min((pageIndex + 1) * pageSize, rows.length)} / ${rows.length}` : "Không có kết quả"} · Kiến thức đang tạm dừng sẽ không được dùng.</span><div className="pagination"><Button variant="ghost" size="icon" aria-label="Trang trước" disabled={pageIndex === 0} onClick={() => setPageIndex((value) => Math.max(0, value - 1))}><ChevronLeft size={15} /></Button><span>{pageIndex + 1} / {pageCount}</span><Button variant="ghost" size="icon" aria-label="Trang sau" disabled={pageIndex + 1 >= pageCount} onClick={() => setPageIndex((value) => Math.min(pageCount - 1, value + 1))}><ChevronRight size={15} /></Button></div></div>
    </Card>
    {Array.isArray(data.corpus?.items) && data.corpus.items.length > 0 && <Card className="corpus-card"><CardHeader><div><div className="eyebrow">NGUỒN THAM KHẢO</div><CardTitle>Tài liệu kiến thức</CardTitle></div><span className="readonly-label"><Database size={13} /> Chỉ đọc</span></CardHeader><div className="corpus-list">{data.corpus.items.slice(0, 6).map((item, index) => <div key={index}><Database size={14} /><span>{String(item.file ?? item.title ?? "Tài liệu")}</span><small>{String(item.role ?? "Nguồn tham khảo")}</small></div>)}</div></Card>}
  </div>;
}

function KnowledgeDetail({ item, onBack, onEdit, onChanged, onToast }: {
  item: Definition;
  onBack: () => void;
  onEdit: () => void;
  onChanged: () => void;
  onToast: (message: string) => void;
}) {
  const properties = item.input_schema?.properties ?? {};
  const required = new Set(item.input_schema?.required ?? []);
  const aliases = item.aliases ?? [];
  return <div className="page detail-page"><div className="page-heading"><div><button className="back-button" onClick={onBack}><ArrowLeft size={15} /> Kiến thức của tôi</button><div className="eyebrow">CHI TIẾT KIẾN THỨC</div><h1>{displayName(item)}</h1><p>{item.describe}</p></div><div className="heading-actions"><Button variant="secondary" onClick={onEdit}>Chỉnh sửa</Button><Button onClick={async () => { try { await request("toggleKnowledge", { intent: item.intent, title: displayName(item), enabled: item.assistant_status === "paused" }); onChanged(); } catch (reason) { onToast(reason instanceof Error ? reason.message : "Không cập nhật được."); } }}>{item.assistant_status === "paused" ? "Kích hoạt" : "Tạm dừng"}</Button></div></div>
    <div className="detail-grid"><div className="detail-main">
      <Card><CardHeader><div><div className="eyebrow">CÁCH NGƯỜI DÙNG CÓ THỂ HỎI</div><CardTitle>Câu ví dụ</CardTitle></div></CardHeader><ul className="example-list">{aliases.map((alias, index) => <li key={`${alias}-${index}`}><span className="example-dot" />{alias}</li>)}</ul></Card>
      <Card><CardHeader><div><div className="eyebrow">THÔNG TIN ĐẦU VÀO</div><CardTitle>{Object.keys(properties).length ? `${Object.keys(properties).length} trường` : "Không cần thông tin thêm"}</CardTitle></div></CardHeader>{Object.keys(properties).length ? <div className="detail-fields">{Object.entries(properties).map(([key, property]) => <div className="detail-field" key={key}><div><strong>{property.title ?? item.field_policies?.[key]?.description ?? key}</strong><span>{property.description ?? item.field_policies?.[key]?.description ?? ""}</span></div><div className="detail-field-meta"><span>{typeLabel(property.type)}</span><span>{required.has(key) ? "Cần hỏi" : Object.hasOwn(property, "default") ? `Mặc định: ${String(property.default)}` : "Có thể bỏ qua"}</span></div></div>)}</div> : <div className="empty-state compact"><Check size={17} /><span>Kiến thức này tạo kết quả từ câu yêu cầu.</span></div>}</Card>
    </div><aside className="detail-aside"><Card><CardHeader><div className="eyebrow">THÔNG TIN</div></CardHeader><InfoRow label="Nhóm" value={categoryLabel(item.category)} /><InfoRow label="Trạng thái" value={item.assistant_status === "paused" ? "Tạm dừng" : "Đang hoạt động"} /><InfoRow label="Kiểu kết quả" value={Object.keys(item.render?.templates ?? {}).map(targetLabel).join(", ") || "—"} />{Array.isArray(item.requires) && item.requires.length > 0 && <InfoRow label="Gói LaTeX" value={(item.requires as string[]).join(", ")} />}<InfoRow label="Phiên bản" value={item.definition_version ?? "Tương thích dữ liệu cũ"} /></Card>
      <Card><CardHeader><div className="eyebrow">MẪU KẾT QUẢ</div></CardHeader>{Object.entries(item.render?.templates ?? {}).map(([target, template]) => <div className="detail-template" key={target}><span>{targetLabel(target)}</span><pre>{template}</pre></div>)}</Card></aside></div>
  </div>;
}

function InfoRow({ label, value }: { label: string; value: string }) { return <div className="info-row"><span>{label}</span><strong>{value}</strong></div>; }

function HistoryPage({ data, onChanged, onToast }: { data: Bootstrap; onChanged: () => void; onToast: (message: string) => void }) {
  const versions = data.history?.versions ?? [];
  const restore = async (version: string) => {
    try {
      const result = await request<{ rolled_back?: boolean; cancelled?: boolean }>("rollback", { version });
      if (result.rolled_back) { onToast("Đã phục hồi phiên bản kiến thức."); onChanged(); }
    } catch (reason) { onToast(reason instanceof Error ? reason.message : "Không phục hồi được."); }
  };
  return <div className="page history-page"><div className="page-heading"><div><div className="eyebrow">CÁC THAY ĐỔI</div><h1>Lịch sử hoạt động</h1><p>Xem lại thao tác và phục hồi bản sao kiến thức đã lưu.</p></div></div>
    <div className="history-grid"><Card><CardHeader><div><div className="eyebrow">HOẠT ĐỘNG TRONG EXTENSION</div><CardTitle>Gần đây</CardTitle></div></CardHeader>{data.activity.length ? <div className="timeline">{data.activity.map((item) => <div className="timeline-item" key={item.id}><span className="timeline-icon"><ActivityIcon size={14} /></span><div><strong>{actionLabels[item.action] ?? item.action}</strong><span>{item.title}</span><small>{formatDate(item.at)}</small></div></div>)}</div> : <div className="empty-state"><Clock3 size={19} /><strong>Chưa có hoạt động</strong><span>Các thao tác của bạn sẽ xuất hiện tại đây.</span></div>}</Card>
      <Card><CardHeader><div><div className="eyebrow">BẢN SAO DỮ LIỆU</div><CardTitle>{versions.length} phiên bản</CardTitle></div></CardHeader>{versions.length ? <div className="version-list">{versions.map((version) => <div className="version-row" key={version.version}><span className="version-icon"><Database size={14} /></span><div><strong>{version.active ? "Phiên bản hiện tại" : `Bản sao ${version.version.slice(0, 10)}`}</strong><small>{formatDate(version.created_at)} · {version.valid_hash ? "Đã kiểm tra" : "Không khớp mã kiểm tra"}</small></div><Button variant="secondary" size="sm" disabled={version.active || !version.valid_hash} onClick={() => void restore(version.version)}>Phục hồi</Button></div>)}</div> : <div className="empty-state"><History size={18} /><strong>Chưa có bản sao</strong><span>Bản sao sẽ được tạo trước thay đổi kiến thức.</span></div>}</Card></div>
  </div>;
}

function SettingsPage({ data, service, onRefresh, onToast }: { data: Bootstrap; service?: ServiceStatus; onRefresh: () => void; onToast: (message: string) => void }) {
  const theme = document.documentElement.dataset.theme ?? "dark";
  const classifierReady = service?.classifierAvailable ?? data.system?.classifier_available ?? false;
  const retrievalReady = service?.retrievalModelAvailable ?? data.system?.retrieval_model_available ?? false;
  const pipelineMode = service?.pipelineMode ?? data.system?.pipeline_mode ?? "unavailable";
  const modeLabel = pipelineMode === "phobert_retrieval" ? "PhoBERT + Retrieval" : pipelineMode === "retrieval_only" ? "Chỉ Retrieval" : pipelineMode === "phobert_only" ? "Chỉ PhoBERT" : "Chưa có mô hình NLP";
  const classifierMessage = service?.classifierMessage ?? data.system?.classifier_message ?? "Chưa nhận được trạng thái PhoBERT.";
  const retrievalMessage = service?.retrievalModelMessage ?? data.system?.retrieval_model_message ?? "Chưa nhận được trạng thái retrieval.";
  const indexMessage = service?.retrievalIndexMessage ?? data.system?.retrieval_index_message ?? "Index sẽ được tạo khi có model retrieval.";
  const modelRoot = service?.modelRoot ?? data.system?.model_root ?? "<workspace>/models";
  const updateTheme = async (value: string) => {
    document.documentElement.dataset.theme = value;
    try { await request("setTheme", { theme: value }); onToast("Đã lưu giao diện."); }
    catch (reason) { onToast(reason instanceof Error ? reason.message : "Không lưu được giao diện."); }
  };
  const chooseImport = async () => {
    try {
      const result = await request<{ imported?: boolean; count?: number; cancelled?: boolean }>("importDefinitions");
      if (result.imported) { onToast(`Đã nhập ${result.count} kiến thức.`); onRefresh(); }
    } catch (reason) { onToast(reason instanceof Error ? reason.message : "Không nhập được tệp."); }
  };
  const chooseExport = async () => {
    try {
      const result = await request<{ exported?: boolean; count?: number; cancelled?: boolean }>("exportDefinitions");
      if (result.exported) onToast(`Đã xuất ${result.count} kiến thức.`);
    } catch (reason) { onToast(reason instanceof Error ? reason.message : "Không xuất được tệp."); }
  };
  return <div className="page settings-page"><div className="page-heading"><div><div className="eyebrow">TÙY CHỈNH</div><h1>Cài đặt</h1><p>Quản lý cách hiển thị và trạng thái dịch vụ cục bộ.</p></div></div>
    <div className="settings-grid"><Card><CardHeader><div><div className="eyebrow">GIAO DIỆN</div><CardTitle>Giao diện ứng dụng</CardTitle></div></CardHeader><div className="theme-picker"><button className={theme === "dark" ? "selected" : ""} onClick={() => void updateTheme("dark")}><span className="theme-swatch dark-swatch"><Moon size={16} /></span><strong>Tối</strong><small>Mặc định</small></button><button className={theme === "light" ? "selected" : ""} onClick={() => void updateTheme("light")}><span className="theme-swatch light-swatch"><Sun size={16} /></span><strong>Sáng</strong><small>Dễ đọc ban ngày</small></button></div></Card>
      <Card><CardHeader><div><div className="eyebrow">PYTHON ENVIRONMENT · BACKEND</div><CardTitle>Dịch vụ cục bộ</CardTitle></div><span className={`status-pill ${service?.state === "ready" ? "active" : "paused"}`}><span />{service?.state === "ready" ? "Sẵn sàng" : service?.state === "degraded" ? "Hạn chế" : service?.state === "error" ? "Lỗi thư viện / dịch vụ" : service?.state === "starting" ? "Đang khởi động" : "Cần kiểm tra"}</span></CardHeader><p className="settings-message">{service?.message || data.system?.message || "Đang kiểm tra dịch vụ."}</p><div className="system-facts"><InfoRow label="Python" value={service?.python ?? "Chưa kết nối"} /><InfoRow label="Chế độ pipeline" value={modeLabel} /><InfoRow label="Địa chỉ dịch vụ" value="Loopback · chỉ trên máy này" /></div><div className="settings-actions"><Button variant="secondary" onClick={() => void request("openPythonSettings")}>Chọn Python</Button><Button variant="secondary" onClick={async () => { try { await request("checkBackend"); onRefresh(); onToast("Đã kiểm tra dịch vụ."); } catch (reason) { onToast(reason instanceof Error ? reason.message : "Không kết nối được dịch vụ."); } }}>Kiểm tra lại</Button><Button onClick={async () => { try { await request("restartBackend"); onRefresh(); onToast("Đã khởi động lại dịch vụ."); } catch (reason) { onToast(reason instanceof Error ? reason.message : "Không khởi động lại được."); } }}>Khởi động lại</Button></div></Card>
      <Card><CardHeader><div><div className="eyebrow">NLP MODELS</div><CardTitle>Mô hình và bộ nhớ chỉ mục</CardTitle></div></CardHeader><div className="model-status-list">
        <ModelStatus name="PhoBERT" detail={service?.classifierCheckpointPath ?? data.system?.classifier_checkpoint_path ?? "checkpoints/phobert_classifier"} ready={classifierReady} message={classifierMessage} readyText="Checkpoint hợp lệ" missingText="Chưa khả dụng" />
        <ModelStatus name="Retrieval model" detail={service?.retrievalModelPath ?? data.system?.retrieval_model_path ?? "hf_cache/vietnamese-bi-encoder"} ready={retrievalReady} message={retrievalMessage} readyText="Model hợp lệ" missingText="Chưa khả dụng" />
        <ModelStatus name="Retrieval index" detail={service?.retrievalIndexPath ?? data.system?.retrieval_index_path ?? modelRoot} ready={service?.retrievalIndexAvailable ?? data.system?.retrieval_index_available ?? false} message={indexMessage} readyText="Cache hợp lệ" missingText="Build ở lần hỏi đầu" />
      </div><p className="settings-message">Model root hiện tại: <code>{modelRoot}</code>. Cấu trúc mặc định: <code>checkpoints/phobert_classifier</code>, <code>hf_cache/phobert-base-v2</code> và <code>hf_cache/vietnamese-bi-encoder</code>. Index chưa có sẽ được tạo khi truy hồi lần đầu.</p><div className="settings-actions"><Button variant="secondary" onClick={() => void request("openModelSettings")}>Cấu hình modelRoot</Button></div></Card>
      <Card><CardHeader><div><div className="eyebrow">SAO LƯU VÀ CHIA SẺ</div><CardTitle>Nhập hoặc xuất kiến thức</CardTitle></div></CardHeader><p className="settings-message">Tệp được kiểm tra trước khi nhập. Khi trùng tên, bạn sẽ được hỏi xác nhận và hệ thống tạo bản sao trước khi thay đổi.</p><div className="settings-actions"><Button variant="secondary" onClick={() => void chooseImport()}><Upload size={14} /> Nhập tệp</Button><Button variant="secondary" onClick={() => void chooseExport()}><ArrowDownToLine size={14} /> Xuất tệp</Button></div></Card>
      <Card><CardHeader><div><div className="eyebrow">GỢI Ý KIẾN THỨC</div><CardTitle>Tạo từ mô tả</CardTitle></div><Sparkles size={16} /></CardHeader><p className="settings-message">{data.builder?.message || "Chưa có mô hình tạo đề xuất cục bộ. Biểu mẫu tạo kiến thức thủ công vẫn hoạt động."}</p><span className="capability-pill">{data.builder?.suggestions_available ? "Có thể gợi ý" : "Biểu mẫu thủ công"}</span></Card>
      <Card><CardHeader><div><div className="eyebrow">TỐI ƯU RETRIEVAL THEO MỐC</div><CardTitle>Sinh Train/Val & Tự học (Mốc 10)</CardTitle></div><Sparkles size={16} /></CardHeader>
        {(() => {
          const milestone = data.system?.milestone;
          const pending = milestone?.pending_count ?? 0;
          const threshold = milestone?.threshold ?? 10;
          const pct = Math.min(100, Math.round((pending / Math.max(1, threshold)) * 100));
          const last = milestone?.last_report;
          const beforeTop1 = last?.before_val?.val_top1;
          const afterTop1 = last?.after_val?.val_top1;
          const learnedAliasCount = data.definitions.reduce((acc, item) => (
            acc + (Array.isArray(item.learned_aliases) ? item.learned_aliases.length : 0)
          ), 0);
          return <>
            <p className="settings-message">Kiến thức mới chạy tức thì nhờ Retrieval. Khi tích lũy đủ {threshold} kiến thức mới, hệ thống có thể sinh tập train/val tách biệt (không chạm test.csv) để tinh chỉnh Retrieval.</p>
            <div className="milestone-bar" aria-label={`Tiến độ mốc: ${pending}/${threshold}`}><span style={{ width: `${pct}%` }} /></div>
            <div className="system-facts">
              <InfoRow label="Chờ huấn luyện" value={milestone?.training_running ? `${pending} / ${threshold} (đang tối ưu…)` : `${pending} / ${threshold} kiến thức mới`} />
              <InfoRow label="Phản hồi đã chọn · Alias" value={`${milestone?.feedback_count ?? 0} lượt · ${learnedAliasCount} alias`} />
              {last && typeof beforeTop1 === "number" && (
                <InfoRow
                  label="Lần tối ưu gần nhất"
                  value={`Val Top-1: ${(beforeTop1 * 100).toFixed(1)}% → ${((afterTop1 ?? beforeTop1) * 100).toFixed(1)}% (${last.device ?? "cpu"}${last.deployed ? " · đã nạp" : " · giữ cũ"})`}
                />
              )}
              {milestone?.last_error && <InfoRow label="Lỗi gần nhất" value={milestone.last_error} />}
            </div>
            <div className="settings-actions">
              <Button disabled={milestone?.training_running === true} onClick={async () => {
                try {
                  await request("triggerMilestoneTraining", { force: true });
                  onToast("Đã khởi chạy sinh train/val & tối ưu Retrieval.");
                  onRefresh();
                } catch (reason) {
                  onToast(reason instanceof Error ? reason.message : "Không thể chạy tối ưu Retrieval.");
                }
              }}>{milestone?.training_running ? "Đang tối ưu…" : "Sinh Train/Val & Tối ưu ngay"}</Button>
            </div>
          </>;
        })()}
      </Card>
      <Card><CardHeader><div><div className="eyebrow">ADVANCED SETTINGS</div><CardTitle>Cấu hình nâng cao & Comment</CardTitle></div><Settings2 size={16} /></CardHeader><p className="settings-message">Tùy chỉnh tiền tố comment cho từng ngôn ngữ (ví dụ Python `#`, Rust `//`, SQL `--`), chế độ Output Panel và Python venv.</p><div className="settings-actions"><Button variant="secondary" onClick={() => void request("openCommentSettings")}>Cấu hình comment ngôn ngữ</Button><Button variant="secondary" onClick={() => void request("openAdvancedSettings")}>Mở tất cả cài đặt</Button></div></Card>
    </div>
  </div>;
}

function ModelStatus({ name, detail, ready, message, readyText, missingText }: { name: string; detail: string; ready: boolean; message: string; readyText: string; missingText: string }) {
  return <div className="model-status-row"><span className={`runtime-dot ${ready ? "ready" : "missing"}`} /><div><strong>{name}</strong><small>{detail}</small><small>{message}</small></div><span className={`model-state ${ready ? "ready" : "missing"}`}>{ready ? readyText : missingText}</span></div>;
}

function ActivityRow({ item }: { item: Activity }) { return <div className="activity-row"><span className="activity-mark"><Check size={13} /></span><div><strong>{actionLabels[item.action] ?? item.action}</strong><span>{item.title}</span></div><small>{formatDate(item.at)}</small></div>; }
function categoryLabel(category?: string): string { return categories.find(([key]) => key === category)?.[1] ?? category ?? "Chưa phân nhóm"; }
function targetLabel(target: string): string {
  if (target === "auto") return "Tự động theo ngữ cảnh";
  return universalTargets.find(([key]) => key === target)?.[1]
    ?? (target === "latex" ? "LaTeX" : target === "mathjax" ? "Công thức toán" : target === "obsidian" ? "Markdown" : target);
}
function typeLabel(type?: string): string { return type === "number" ? "Số" : type === "integer" ? "Số nguyên" : type === "boolean" ? "Có / Không" : type === "object" ? "Nhóm thông tin" : type === "array" ? "Danh sách" : "Văn bản"; }

