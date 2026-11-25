import { useState } from "react";
import { ArrowLeft, ArrowRight, Check, FileText, Plus, Sparkles, Trash2 } from "lucide-react";
import { request } from "../bridge";
import { Button } from "../components/ui/Button";
import { Card, CardHeader, CardTitle } from "../components/ui/Card";
import { FormField, Input, Select, Textarea } from "../components/ui/Field";
import type { BuilderDraft, BuilderField, BuilderSegment, CategoryOption, Definition } from "../types";

const steps = ["Mô tả", "Thông tin", "Kết quả", "Kiểm tra"];
export const categories: Array<[string, string]> = [
  ["general", "Phổ thông / Text"],
  ["python", "Python"],
  ["shell", "Shell / Linux"],
  ["rust", "Rust"],
  ["cpp", "C / C++"],
  ["java", "Java"],
  ["javascript", "JavaScript / TypeScript"],
  ["css", "CSS / HTML"],
  ["sysadmin", "Cài đặt / Hệ thống"],
  ["office", "Văn phòng / Phím tắt"],
  ["tex_doc", "LaTeX: Tài liệu"],
  ["tex_struct", "LaTeX: Cấu trúc"],
  ["tex_text", "LaTeX: Văn bản"],
  ["tex_lists", "LaTeX: Danh sách"],
  ["tex_tables", "LaTeX: Bảng"],
  ["tex_figures", "LaTeX: Hình ảnh"],
  ["tex_math", "LaTeX: Toán học"],
  ["math_shared", "Ký hiệu toán"],
  ["obsidian", "Markdown / Obsidian"],
  ["tex_misc", "Khác"],
];

export const universalTargets: Array<[string, string]> = [
  ["text", "Văn bản thuần / Ghi chú (.txt)"],
  ["python", "Python (.py)"],
  ["shell", "Shell / Linux (.sh)"],
  ["rust", "Rust (.rs)"],
  ["cpp", "C / C++ (.cpp)"],
  ["java", "Java (.java)"],
  ["javascript", "JavaScript (.js)"],
  ["typescript", "TypeScript (.ts)"],
  ["css", "CSS (.css)"],
  ["html", "HTML (.html)"],
  ["sql", "SQL (.sql)"],
  ["lua", "Lua (.lua)"],
  ["markdown", "Markdown (.md)"],
  ["obsidian", "Obsidian Markdown"],
  ["latex", "LaTeX (.tex)"],
  ["mathjax", "MathJax (Công thức toán)"],
];

type BuilderMode = "quick" | "advanced";

function slugKey(value: string): string {
  const ascii = value.toLocaleLowerCase().replace(/đ/g, "d").normalize("NFKD").replace(/[\u0300-\u036f]/g, "");
  const key = ascii.replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  return /^[a-z]/.test(key) ? key : `field_${key || "value"}`;
}

function definitionToDraft(definition: Definition): BuilderDraft {
  const schema = definition.input_schema ?? {};
  const properties = schema.properties ?? {};
  const required = new Set(schema.required ?? []);
  const fields: BuilderField[] = Object.entries(properties).map(([key, property]) => {
    const policy = definition.field_policies?.[key];
    const valueType = property.type ?? policy?.value_type ?? "string";
    if (!["string", "number", "integer", "boolean"].includes(valueType)) {
      throw new Error(`Trường “${property.title ?? key}” cần chỉnh sửa nâng cao, chưa thể mở bằng biểu mẫu này.`);
    }
    return {
      key,
      label: property.title ?? policy?.description ?? key,
      value_type: valueType as BuilderField["value_type"],
      required: required.has(key),
      has_default: Object.hasOwn(property, "default") || policy?.on_missing?.action === "use_default",
      default: Object.hasOwn(property, "default") ? property.default : policy?.on_missing?.value ?? "",
    };
  });
  const templates = definition.render?.templates ?? {};
  const preferredOrder = ["text", "python", "shell", "rust", "cpp", "java", "javascript", "latex", "mathjax", "obsidian"];
  const target = preferredOrder.find((item) => typeof templates[item] === "string")
    ?? Object.keys(templates).find((item) => typeof templates[item] === "string")
    ?? "text";
  const template = templates[target];
  if (typeof template !== "string") throw new Error("Kiến thức này chưa có mẫu kết quả được hỗ trợ để chỉnh sửa bằng biểu mẫu.");
  const segments: BuilderSegment[] = [];
  const token = /\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\|\s*latex_escape)?\s*\}\}/g;
  let cursor = 0;
  for (const match of template.matchAll(token)) {
    const offset = match.index ?? 0;
    const key = match[1];
    if (!fields.some((field) => field.key === key)) throw new Error("Mẫu kết quả có thành phần nâng cao không thể chỉnh sửa an toàn bằng biểu mẫu.");
    if (offset > cursor) segments.push({ kind: "text", value: template.slice(cursor, offset) });
    segments.push({ kind: "field", value: key });
    cursor = offset + match[0].length;
  }
  const leftover = template.slice(cursor);
  if (leftover.includes("{{") || leftover.includes("{%")) throw new Error("Mẫu kết quả có thành phần nâng cao không thể chỉnh sửa an toàn bằng biểu mẫu.");
  if (leftover) segments.push({ kind: "text", value: leftover });
  return {
    intent: definition.intent,
    name: definition.display_name ?? definition.intent,
    description: definition.describe ?? "",
    category: definition.category ?? "tex_misc",
    examples: definition.aliases ?? [],
    fields,
    segments,
    target,
    base_definition: definition,
  };
}

function newDraft(mode: BuilderMode = "quick"): BuilderDraft {
  if (mode === "quick") {
    return {
      name: "",
      description: "",
      category: "general",
      examples: [""],
      fields: [],
      segments: [{ kind: "text", value: "" }],
      target: "text",
    };
  }
  return {
    name: "",
    description: "",
    category: "python",
    examples: [""],
    fields: [],
    segments: [{ kind: "text", value: "" }],
    target: "python",
  };
}

export function Builder({ initialDefinition, categoryOptions, onBack, onPublished }: {
  initialDefinition?: Definition;
  categoryOptions?: CategoryOption[];
  onBack: () => void;
  onPublished: (intent: string) => void;
}) {
  const [draft, setDraft] = useState<BuilderDraft>(() => initialDefinition ? definitionToDraft(initialDefinition) : newDraft("quick"));
  const [mode, setMode] = useState<BuilderMode>(() => {
    if (!initialDefinition) return "quick";
    const parsed = definitionToDraft(initialDefinition);
    return parsed.category === "general" && parsed.target === "text" && parsed.fields.length === 0
      ? "quick"
      : "advanced";
  });
  const [customCategory, setCustomCategory] = useState("");
  const [step, setStep] = useState(0);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [preview, setPreview] = useState<Record<string, unknown> | null>(null);
  const [compiled, setCompiled] = useState<Definition | null>(null);
  const [testValues, setTestValues] = useState<Record<string, unknown>>({});
  const isEditing = Boolean(draft.intent);

  const mergedCategories: CategoryOption[] = (() => {
    const map = new Map<string, string>(categories);
    for (const option of categoryOptions ?? []) {
      map.set(option.value, option.label);
    }
    if (draft.category && !map.has(draft.category)) {
      map.set(draft.category, draft.category);
    }
    return Array.from(map.entries()).map(([value, label]) => ({ value, label }));
  })();

  const switchMode = (nextMode: BuilderMode) => {
    setMode(nextMode);
    setMessage("");
    setPreview(null);
    setCompiled(null);
    if (nextMode === "quick") {
      setDraft((current) => ({
        ...current,
        category: "general",
        target: "text",
        fields: [],
        segments: current.segments.filter((segment) => segment.kind === "text").length > 0
          ? current.segments.filter((segment) => segment.kind === "text")
          : [{ kind: "text", value: "" }],
      }));
    } else if (draft.category === "general" && draft.target === "text") {
      setDraft((current) => ({
        ...current,
        category: "python",
        target: "python",
      }));
    }
  };

  const updateDraft = (update: Partial<BuilderDraft>) => {
    setDraft((current) => ({ ...current, ...update }));
    setPreview(null);
    setCompiled(null);
    setMessage("");
  };

  const ensureCompiled = async (): Promise<Definition> => {
    const result = await request<{ valid: boolean; errors?: Array<{ message?: string }>; definition: Definition }>("compileBuilder", draft as unknown as Record<string, unknown>);
    if (!result.valid) throw new Error(result.errors?.map((error) => error.message).join("; ") || "Thông tin chưa hợp lệ.");
    setCompiled(result.definition);
    return result.definition;
  };

  const runPreview = async () => {
    setBusy(true);
    setMessage("");
    try {
      const definition = await ensureCompiled();
      const checked = await request<{ valid: boolean; errors?: unknown[] }>("validateDefinition", { definition });
      if (!checked.valid) throw new Error(JSON.stringify(checked.errors));
      const result = await request<Record<string, unknown>>("previewDefinition", {
        definition,
        values: testValues,
        target: draft.target,
      });
      setPreview(result);
      if (result.status !== "ok") throw new Error("Hãy bổ sung các giá trị kiểm thử còn thiếu hoặc sửa thông tin không hợp lệ.");
      setMessage("Kiểm tra thành công. Bạn có thể xem lại trước khi lưu.");
    } catch (error) {
      setPreview(null);
      setMessage(error instanceof Error ? error.message : "Không thể kiểm tra kiến thức.");
    } finally {
      setBusy(false);
    }
  };

  const publish = async () => {
    if (!preview || preview.status !== "ok" || !compiled) {
      setMessage("Hãy chạy kiểm tra thành công trước khi lưu.");
      return;
    }
    setBusy(true);
    try {
      const validation = await request<{ valid: boolean; errors?: unknown[] }>("validateDefinition", { definition: compiled });
      if (!validation.valid) throw new Error(JSON.stringify(validation.errors));
      const result = await request<{ published?: boolean; cancelled?: boolean; intent: string }>("publishDefinition", {
        definition: compiled,
        title: draft.name,
        overwrite: isEditing,
      });
      if (!result.cancelled) onPublished(result.intent);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Không thể lưu kiến thức.");
    } finally {
      setBusy(false);
    }
  };

  const validateStep = () => {
    if (step === 0) {
      if (!draft.name.trim() || !draft.description.trim() || !draft.examples.some((value) => value.trim())) {
        setMessage("Nhập tên, mô tả và ít nhất một câu ví dụ.");
        return false;
      }
    }
    if (step === 1) {
      const keys = draft.fields.map((field) => field.key);
      if (new Set(keys).size !== keys.length || draft.fields.some((field) => !field.label.trim() || !field.key)) {
        setMessage("Mỗi trường cần có tên riêng biệt.");
        return false;
      }
    }
    if (step === 2 && (!draft.segments.length || draft.segments.some((item) => item.kind === "text" && !item.value))) {
      setMessage("Thêm nội dung hoặc trường vào kết quả.");
      return false;
    }
    setMessage("");
    return true;
  };

  const addField = () => {
    const field: BuilderField = {
      key: `field_${draft.fields.length + 1}`,
      label: "Thông tin mới",
      value_type: "string",
      required: true,
      has_default: false,
      default: "",
    };
    updateDraft({ fields: [...draft.fields, field] });
  };

  const updateField = (index: number, patch: Partial<BuilderField>) => {
    const next = draft.fields.map((field, current) => {
      if (index !== current) return field;
      const updated = { ...field, ...patch };
      if (patch.label !== undefined && (!field.key || field.key.startsWith("field_") || field.key === slugKey(field.label))) {
        updated.key = slugKey(patch.label);
      }
      return updated;
    });
    updateDraft({ fields: next });
  };

  const addSegment = (kind: "text" | "field") => {
    if (kind === "field" && draft.fields.length === 0) {
      setMessage("Thêm thông tin đầu vào trước, hoặc chọn đoạn văn bản tĩnh.");
      return;
    }
    updateDraft({ segments: [...draft.segments, { kind, value: kind === "field" ? draft.fields[0].key : "" }] });
  };

  const updateSegment = (index: number, patch: Partial<BuilderSegment>) => {
    updateDraft({ segments: draft.segments.map((segment, current) => current === index ? { ...segment, ...patch } : segment) });
  };

  const nextStep = () => {
    if (!validateStep()) return;
    if (step === 0 && mode === "quick") {
      setStep(2);
      return;
    }
    setStep((current) => current + 1);
  };

  const prevStep = () => {
    setMessage("");
    if (step === 0) {
      onBack();
      return;
    }
    if (step === 2 && mode === "quick") {
      setStep(0);
      return;
    }
    setStep((current) => current - 1);
  };

  const previewOutput = preview?.outputs && typeof preview.outputs === "object"
    ? (preview.outputs as Record<string, unknown>)[draft.target]
    : undefined;

  return <div className="builder-page">
    <div className="page-heading builder-page-heading">
      <div><button className="back-button" onClick={onBack}><ArrowLeft size={15} /> Quay lại</button>
        <div className="eyebrow">TẠO KIẾN THỨC</div><h1>{isEditing ? "Chỉnh sửa kiến thức" : "Tạo kiến thức mới"}</h1>
        <p>Chọn chế độ Nhanh cho ghi chú phổ thông, hoặc Chuyên sâu cho code, LaTeX và tham số.</p></div>
      <span className="draft-pill">{isEditing ? "Đang chỉnh sửa" : mode === "quick" ? "Chế độ Nhanh" : "Chế độ Chuyên sâu"}</span>
    </div>

    <div className="wizard-stepper" aria-label="Các bước tạo kiến thức">
      {steps.map((label, index) => <button key={label} className={`step-item ${step === index ? "current" : step > index ? "complete" : ""}`} onClick={() => index < step && !(mode === "quick" && index === 1) && setStep(index)} aria-current={step === index ? "step" : undefined}>
        <span className="step-index">{step > index ? <Check size={14} /> : index + 1}</span><span>{label}</span>
      </button>)}
    </div>

    <div className="builder-grid">
      <Card className="builder-card">
        <CardHeader><div><div className="eyebrow">BƯỚC {step + 1} / 4</div><CardTitle>{steps[step]}</CardTitle></div></CardHeader>
        <div className="builder-content">
          {step === 0 && <div className="form-stack">
            <div className="mode-switch" role="group" aria-label="Chế độ tạo kiến thức">
              <button type="button" className={mode === "quick" ? "active" : ""} onClick={() => switchMode("quick")}>
                <strong>Nhanh (Phổ thông / Text)</strong>
                <small>Dùng cho setup phần mềm, phím tắt Word, mẹo Linux… Bỏ qua nhóm & tham số.</small>
              </button>
              <button type="button" className={mode === "advanced" ? "active" : ""} onClick={() => switchMode("advanced")}>
                <strong>Chuyên sâu (Code / LaTeX / Tham số)</strong>
                <small>Tự chọn ngôn ngữ (Python, Rust, Shell…), nhóm kiến thức và biến thay thế.</small>
              </button>
            </div>
            <FormField label="Tên kiến thức"><Input autoFocus value={draft.name} onChange={(event) => updateDraft({ name: event.target.value })} placeholder="Ví dụ: Cài đặt Sunshine hoặc Hàm Fibonacci Python" /></FormField>
            <FormField label="Mô tả" hint="Mô tả ngắn gọn công dụng để hệ thống phân biệt rõ với kiến thức khác."><Textarea rows={3} value={draft.description} onChange={(event) => updateDraft({ description: event.target.value })} placeholder="Hướng dẫn mở cổng và cấu hình Sunshine stream game qua LAN." /></FormField>
            {mode === "advanced" && <>
              <FormField label="Nhóm kiến thức">
                <Select
                  value={mergedCategories.some((c) => c.value === draft.category) ? draft.category : "__custom__"}
                  onChange={(event) => {
                    const value = event.target.value;
                    if (value === "__custom__") {
                      updateDraft({ category: customCategory || "custom_topic" });
                    } else {
                      updateDraft({ category: value });
                    }
                  }}
                >
                  {mergedCategories.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                  <option value="__custom__">+ Tự nhập nhóm mới…</option>
                </Select>
              </FormField>
              {(!mergedCategories.some((c) => c.value === draft.category) || customCategory !== "") && (
                <FormField label="Mã nhóm tùy chỉnh (tiếng Anh không dấu, vd: docker, music, git)">
                  <Input
                    value={customCategory || draft.category}
                    onChange={(event) => {
                      const raw = slugKey(event.target.value);
                      setCustomCategory(event.target.value);
                      updateDraft({ category: raw });
                    }}
                    placeholder="Ví dụ: docker"
                  />
                </FormField>
              )}
            </>}
            <FormField label="Câu ví dụ kích hoạt" hint="Mỗi dòng là một cách bạn sẽ gõ khi hỏi hoặc viết comment `# A: ...`."><Textarea rows={4} value={draft.examples.join("\n")} onChange={(event) => updateDraft({ examples: event.target.value.split("\n") })} placeholder={"Setup Sunshine trên Linux\nCấu hình stream Moonlight qua LAN\nMở port cho Sunshine"} /></FormField>
          </div>}

          {step === 1 && <div className="form-stack">
            <div className="section-intro"><div><strong>Thông tin người dùng cần cung cấp</strong><p>Chọn kiểu dữ liệu và việc cần làm nếu thông tin bị thiếu.</p></div><Button variant="secondary" size="sm" onClick={addField}><Plus size={14} /> Thêm trường</Button></div>
            {draft.fields.length === 0 && <div className="empty-state compact"><FileText size={18} /><span>Kiến thức này không cần thông tin đầu vào.</span></div>}
            {draft.fields.map((field, index) => <div className="field-editor" key={`${field.key}-${index}`}>
              <div className="field-editor-top"><FormField label="Tên hiển thị" className="grow"><Input value={field.label} onChange={(event) => updateField(index, { label: event.target.value })} placeholder="Ví dụ: Tử số hoặc Tên hàm" /></FormField>
                <Button variant="ghost" size="icon" aria-label={`Xóa ${field.label}`} onClick={() => updateDraft({ fields: draft.fields.filter((_, i) => i !== index) })}><Trash2 size={15} /></Button></div>
              <div className="field-editor-options">
                <FormField label="Kiểu dữ liệu"><Select value={field.value_type} onChange={(event) => updateField(index, { value_type: event.target.value as BuilderField["value_type"] })}>
                  <option value="string">Văn bản</option><option value="number">Số</option><option value="integer">Số nguyên</option><option value="boolean">Có / Không</option>
                </Select></FormField>
                <FormField label="Khi thiếu thông tin"><Select value={field.has_default ? "default" : field.required ? "ask" : "omit"} onChange={(event) => {
                  const behavior = event.target.value;
                  updateField(index, { required: behavior === "ask", has_default: behavior === "default", default: behavior === "default" ? field.default : "" });
                }}><option value="ask">Hỏi người dùng</option><option value="default">Dùng giá trị mặc định</option><option value="omit">Bỏ qua</option></Select></FormField>
              </div>
              {field.has_default && <FormField label="Giá trị mặc định"><Input value={String(field.default ?? "")} onChange={(event) => updateField(index, { default: event.target.value })} placeholder="Nhập giá trị mặc định có căn cứ" /></FormField>}
            </div>)}
          </div>}

          {step === 2 && <div className="form-stack">
            <FormField label="Kiểu kết quả"><Select value={draft.target} onChange={(event) => updateDraft({ target: event.target.value })}>
              {universalTargets.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </Select></FormField>
            <div><div className="field-label">{mode === "quick" ? "Nội dung kiến thức trả ra Output" : "Mẫu kết quả"}</div><p className="field-hint">{mode === "quick" ? "Nhập đoạn văn bản, các bước setup hoặc phím tắt cần in ra khi kích hoạt." : "Thêm đoạn chữ/code và chèn các biến thông tin (nếu có)."}</p>
              <div className="segment-list">{draft.segments.map((segment, index) => <div className="segment-row" key={`${index}-${segment.kind}`}>
                <span className={`segment-kind ${segment.kind}`}>{segment.kind === "text" ? "Nội dung" : "Thông tin"}</span>
                {segment.kind === "text" ? <Textarea rows={mode === "quick" ? 6 : 3} value={segment.value} onChange={(event) => updateSegment(index, { value: event.target.value })} placeholder="Nhập nội dung text, đoạn code hoặc lệnh shell trả về..." /> : <Select value={segment.value} onChange={(event) => updateSegment(index, { value: event.target.value })}>{draft.fields.map((field) => <option key={field.key} value={field.key}>{field.label}</option>)}</Select>}
                <Button variant="ghost" size="icon" aria-label="Xóa thành phần" onClick={() => updateDraft({ segments: draft.segments.filter((_, i) => i !== index) })}><Trash2 size={14} /></Button>
              </div>)}</div>
              <div className="segment-actions"><Button variant="secondary" size="sm" onClick={() => addSegment("text")}><Plus size={14} /> Thêm đoạn nội dung</Button>{mode === "advanced" && <Button variant="secondary" size="sm" onClick={() => addSegment("field")}><Plus size={14} /> Chèn thông tin</Button>}</div>
            </div>
          </div>}

          {step === 3 && <div className="form-stack">
            <div className="review-summary"><div><span>Tên</span><strong>{draft.name || "Chưa đặt tên"}</strong></div><div><span>Nhóm</span><strong>{mergedCategories.find((item) => item.value === draft.category)?.label ?? draft.category}</strong></div><div><span>Định dạng · Biến</span><strong>{universalTargets.find(([key]) => key === draft.target)?.[1] ?? draft.target} · {draft.fields.length || "Không cần"}</strong></div></div>
            <div className="test-values"><div><strong>Thử kiến thức</strong><p>Giá trị ở đây chỉ dùng để xem trước, không được lưu làm mặc định.</p></div>
              {draft.fields.length > 0 && draft.fields.map((field) => <FormField key={field.key} label={field.label}>
                {field.value_type === "boolean"
                  ? <Select value={String(testValues[field.key] ?? field.default ?? "")} onChange={(event) => setTestValues((current) => ({ ...current, [field.key]: event.target.value === "" ? "" : event.target.value === "true" }))}><option value="">Chọn giá trị</option><option value="true">Có</option><option value="false">Không</option></Select>
                  : <Input value={String(testValues[field.key] ?? field.default ?? "")} type={field.value_type === "number" || field.value_type === "integer" ? "number" : "text"} onChange={(event) => setTestValues((current) => ({ ...current, [field.key]: event.target.value }))} placeholder={field.required ? "Nhập giá trị thử" : "Có thể để trống"} />}
              </FormField>)}
            </div>
            <div className="review-output"><div className="section-intro"><div><strong>Kết quả xem trước</strong><p>{preview ? "Được tạo từ Backend hiện tại." : "Chưa chạy kiểm tra."}</p></div><Button onClick={() => void runPreview()} disabled={busy}><Sparkles size={14} /> {busy ? "Đang kiểm tra" : "Kiểm tra & xem trước"}</Button></div>
              {typeof previewOutput === "string" ? <pre className="output-preview">{previewOutput}</pre> : preview && <pre className="output-preview error-preview">{JSON.stringify(preview.errors ?? preview, null, 2)}</pre>}
            </div>
          </div>}
        </div>
        <footer className="wizard-footer">
          <Button variant="secondary" onClick={prevStep}><ArrowLeft size={14} /> {step === 0 ? "Hủy" : "Trước"}</Button>
          <div className="wizard-footer-end">{message && <span className={`inline-message ${preview?.status === "ok" ? "success" : ""}`} role="status">{message}</span>}
            {step < steps.length - 1 ? <Button onClick={nextStep}>Tiếp theo <ArrowRight size={14} /></Button>
              : <Button onClick={() => void publish()} disabled={busy || !preview || preview.status !== "ok"}><Check size={14} /> {busy ? "Đang lưu" : isEditing ? "Xác nhận cập nhật" : "Xác nhận lưu"}</Button>}
          </div>
        </footer>
      </Card>

      <aside className="builder-side">
        <Card><CardHeader><div className="eyebrow">GỢI Ý</div><CardTitle>Lưu và tự động học</CardTitle></CardHeader><div className="builder-tip"><Sparkles size={16} /><p>Kiến thức 1–9 khả dụng tức thì qua Retrieval. Đủ mốc 10 kiến thức mới, hệ thống tự sinh train/val để tinh chỉnh chỉ mục.</p></div></Card>
        <Card><CardHeader><div className="eyebrow">QUY TRÌNH AN TOÀN</div><CardTitle>Kiểm tra trước khi lưu</CardTitle></CardHeader><ol className="safety-steps"><li>Thông tin được kiểm tra theo định nghĩa dùng chung.</li><li>Kết quả xem trước được tạo bởi Backend.</li><li>Chỉ lưu khi bạn xác nhận bước cuối.</li></ol></Card>
      </aside>
    </div>
  </div>;
}

export { definitionToDraft, newDraft, slugKey };
