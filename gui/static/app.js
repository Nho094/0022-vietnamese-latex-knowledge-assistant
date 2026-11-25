/* Local UI client. Business rules and persistence stay in the FastAPI backend. */
(() => {
  "use strict";

  const API = "/knowledge";
  const WIZARD_STEPS = [
    { title: "Thông tin chung", subtitle: "Đặt tên và mô tả kiến thức." },
    { title: "Cấu trúc dữ liệu", subtitle: "Khai báo field động, object và array lồng nhau." },
    { title: "Extraction policies", subtitle: "Chọn cách lấy dữ liệu và hành vi khi thiếu field." },
    { title: "Transform", subtitle: "Biến đổi dữ liệu bằng JSONata." },
    { title: "Render templates", subtitle: "Khai báo nhiều output bằng Jinja2." },
    { title: "Validate & Publish", subtitle: "Kiểm tra contract trước khi gửi backend." },
  ];
  const JSON_TYPES = ["string", "number", "integer", "boolean", "object", "array"];
  const EXTRACTION_METHODS = ["manual", "regex", "parser", "path", "model"];
  const MISSING_ACTIONS = ["ask_user", "use_default", "use_value", "omit", "error"];
  const state = {
    definitions: [],
    corpora: [],
    step: 0,
    editing: false,
    previewedDefinition: "",
    definition: freshDefinition(),
  };

  function freshDefinition() {
    return {
      intent: "",
      describe: "",
      aliases: [],
      category: "",
      domain: "latex",
      definition_version: "2.0",
      input_schema: { type: "object", properties: {}, required: [], additionalProperties: false },
      field_policies: {},
      transform: { engine: "jsonata", expression: "$", mode: "identity" },
      render: { engine: "jinja2", context: "normalized_input", templates: { latex: "" } },
      slots: {},
      templates: { latex: "", python: "", cpp: "" },
      kind: "snippet",
      requires: [],
      compilable: false,
    };
  }

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);

  function api(path, options = {}) {
    return request(`${API}${path}`, options);
  }

  async function request(path, options = {}) {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = body.detail;
      const message = typeof detail === "string" ? detail : JSON.stringify(detail || body);
      throw new Error(message || `HTTP ${response.status}`);
    }
    return body;
  }

  function toast(message, kind = "") {
    const region = $("#toast-region");
    const node = document.createElement("div");
    node.className = `toast ${kind}`;
    node.textContent = message;
    region.append(node);
    window.setTimeout(() => node.remove(), 4200);
  }

  function setView(view) {
    $("#dashboard-view").classList.toggle("hidden", view !== "dashboard");
    $("#builder-view").classList.toggle("hidden", view !== "builder");
    $$(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.view === view));
    if (view === "dashboard") loadDashboard();
    if (view === "builder") renderWizard();
  }

  async function loadDashboard() {
    try {
      const [registry, corpus] = await Promise.all([
        api("/definitions"),
        api("/corpus"),
      ]);
      state.definitions = registry.definitions || [];
      state.corpora = corpus.corpora || [];
      $("#definition-count").textContent = registry.count ?? state.definitions.length;
      $("#corpus-count").textContent = state.corpora.length;
      fillFilter("#category-filter", registry.categories || [], "Tất cả category");
      fillFilter("#domain-filter", registry.domains || [], "Tất cả domain");
      renderDefinitions();
      renderCorpora();
    } catch (error) {
      $("#definition-rows").innerHTML = `<tr><td colspan="5" class="empty-cell">Không tải được registry: ${esc(error.message)}</td></tr>`;
      $("#corpus-list").innerHTML = `<div class="empty-cell">Không tải được catalog: ${esc(error.message)}</div>`;
    }
  }

  function fillFilter(selector, values, initial) {
    const select = $(selector);
    const selected = select.value;
    select.innerHTML = `<option value="">${esc(initial)}</option>${values.map((value) => `<option value="${esc(value)}">${esc(value)}</option>`).join("")}`;
    select.value = values.includes(selected) ? selected : "";
  }

  function renderDefinitions() {
    const search = $("#search-input").value.trim().toLocaleLowerCase();
    const category = $("#category-filter").value;
    const domain = $("#domain-filter").value;
    const rows = state.definitions.filter((item) => {
      const haystack = [item.intent, item.describe, ...(item.aliases || [])].join(" ").toLocaleLowerCase();
      return (!search || haystack.includes(search))
        && (!category || item.category === category)
        && (!domain || item.domain === domain);
    });
    $("#visible-count").textContent = `${rows.length} definitions`;
    $("#definition-rows").innerHTML = rows.length ? rows.map((item) => `
      <tr>
        <td><div class="intent-name">${esc(item.intent)}</div><div class="intent-desc">${esc(item.describe || "Chưa có mô tả")}</div></td>
        <td><span class="category-tag">${esc(item.category || "—")}<small>${esc(item.domain || "domain chưa khai báo")}</small></span></td>
        <td><span class="version-pill">v${esc(item.definition_version || "legacy")}</span></td>
        <td><span class="valid-pill">✓ Schema OK</span></td>
        <td><button class="row-open" data-intent="${esc(item.intent)}" aria-label="Mở ${esc(item.intent)}">›</button></td>
      </tr>`).join("") : `<tr><td colspan="5" class="empty-cell">Không tìm thấy definition phù hợp.</td></tr>`;
    $$(".row-open").forEach((button) => button.addEventListener("click", () => openDefinition(button.dataset.intent)));
  }

  function renderCorpora() {
    $("#corpus-list").innerHTML = state.corpora.length ? state.corpora.map((item) => `
      <article class="corpus-card"><strong>${esc(item.file)}</strong><span>${esc(item.role || "knowledge corpus")}</span><span>${esc(item.record_path || "records")}</span></article>`).join("") : `<div class="empty-cell">Catalog chưa khai báo corpus.</div>`;
  }

  async function openDefinition(intent) {
    try {
      const definition = await api(`/definitions/${encodeURIComponent(intent)}`);
      state.definition = normalizeDefinition(definition);
      state.editing = true;
      state.previewedDefinition = "";
      state.step = 0;
      $("#builder-title").textContent = `Chỉnh sửa · ${intent}`;
      setView("builder");
    } catch (error) {
      toast(error.message, "error");
    }
  }

  function normalizeDefinition(definition) {
    const result = { ...freshDefinition(), ...structuredClone(definition) };
    result.input_schema ||= { type: "object", properties: {}, required: [] };
    result.input_schema.properties ||= {};
    result.field_policies ||= {};
    result.transform ||= { engine: "jsonata", expression: "$" };
    result.render ||= { engine: "jinja2", templates: {} };
    result.render.templates ||= {};
    return result;
  }

  function startNewDefinition() {
    state.definition = freshDefinition();
    state.editing = false;
    state.previewedDefinition = "";
    state.step = 0;
    $("#builder-title").textContent = "Tạo kiến thức mới";
    setView("builder");
  }

  function renderWizard() {
    renderWizardRail();
    const step = WIZARD_STEPS[state.step];
    $("#step-progress").style.width = `${((state.step + 1) / WIZARD_STEPS.length) * 100}%`;
    $("#previous-step").disabled = state.step === 0;
    $("#next-step").textContent = state.step === WIZARD_STEPS.length - 1 ? "Validate & publish" : "Tiếp theo →";
    $("#wizard-message").textContent = "";
    $("#wizard-content").innerHTML = `<div class="wizard-body"><div class="step-kicker">Bước ${state.step + 1} / ${WIZARD_STEPS.length}</div><h2>${esc(step.title)}</h2><p class="step-description">${esc(step.subtitle)}</p>${renderStepContent()}</div>`;
    if (state.step === 1) bindSchemaEditor($("#wizard-content"));
    if (state.step === 2) bindPolicyEditor($("#wizard-content"));
    if (state.step === 4) bindTemplateEditor($("#wizard-content"));
  }

  function renderWizardRail() {
    $("#wizard-steps").innerHTML = WIZARD_STEPS.map((step, index) => `
      <button class="step-item ${index === state.step ? "active" : ""} ${index < state.step ? "done" : ""}" data-step="${index}" ${index > state.step ? "disabled" : ""}>
        <span class="step-number">${index < state.step ? "✓" : index + 1}</span><span>${esc(step.title)}</span>
      </button>`).join("");
    $$(".step-item").forEach((item) => item.addEventListener("click", () => goToStep(Number(item.dataset.step))));
  }

  function renderStepContent() {
    if (state.step === 0) return renderGeneralStep();
    if (state.step === 1) return renderSchemaStep();
    if (state.step === 2) return renderPolicyStep();
    if (state.step === 3) return renderTransformStep();
    if (state.step === 4) return renderRenderStep();
    return renderPublishStep();
  }

  function renderGeneralStep() {
    const definition = state.definition;
    return `<div class="form-grid">
      ${control("Intent ID", "intent", definition.intent, "VD: CUSTOM_TABLE_RENDER", "Tên duy nhất, dùng chữ hoa, số và dấu gạch dưới.")}
      ${control("Category", "category", definition.category, "VD: custom_documents")}
      ${control("Domain", "domain", definition.domain, "VD: latex")}
      ${control("Description", "describe", definition.describe, "Mô tả ngắn bằng tiếng Việt", "", true)}
      ${textareaControl("Aliases", "aliases", (definition.aliases || []).join("\n"), "Mỗi alias trên một dòng", "", true)}
    </div>`;
  }

  function control(label, key, value, placeholder, helper = "", full = false) {
    return `<div class="form-control ${full ? "full" : ""}"><label for="field-${key}">${esc(label)}</label><input id="field-${key}" data-general="${key}" value="${esc(value)}" placeholder="${esc(placeholder)}" />${helper ? `<span class="helper">${esc(helper)}</span>` : ""}</div>`;
  }

  function textareaControl(label, key, value, placeholder, helper = "", full = false) {
    return `<div class="form-control ${full ? "full" : ""}"><label for="field-${key}">${esc(label)}</label><textarea id="field-${key}" data-general="${key}" placeholder="${esc(placeholder)}">${esc(value)}</textarea>${helper ? `<span class="helper">${esc(helper)}</span>` : ""}</div>`;
  }

  function renderSchemaStep() {
    const schema = state.definition.input_schema || { properties: {} };
    const required = new Set(schema.required || []);
    const fields = Object.entries(schema.properties || {}).map(([name, property]) => schemaFieldMarkup(name, { ...property, __required: required.has(name) }, true)).join("");
    return `<div class="schema-toolbar"><div><strong>Root object fields</strong><div class="helper">Cấu trúc có thể lồng object và array không giới hạn theo UI.</div></div><label class="check-control"><input type="checkbox" id="root-additional" ${schema.additionalProperties ? "checked" : ""}/> Cho phép field ngoài schema</label></div>
      <div class="field-list" data-field-list="root">${fields}</div><button class="add-field" data-add-field="root">＋ Thêm field</button>`;
  }

  function schemaFieldMarkup(name = "", schema = { type: "string" }, property = true) {
    const type = JSON_TYPES.includes(schema.type) ? schema.type : "string";
    const options = JSON_TYPES.map((item) => `<option value="${item}" ${item === type ? "selected" : ""}>${item}</option>`).join("");
    const required = property ? `<label class="check-control"><input class="field-required" type="checkbox" ${schema.__required ? "checked" : ""}/> Required</label>` : "";
    const fieldName = property ? `<input class="field-name" value="${esc(name)}" placeholder="field_name" aria-label="Tên field"/>` : `<span class="field-label">Array item schema</span>`;
    return `<article class="schema-field" data-property="${property}">
      <div class="schema-field-header">${fieldName}<select class="field-type" aria-label="Kiểu dữ liệu">${options}</select>${required}<button class="remove-field" type="button" aria-label="Xóa field">×</button></div>
      <div class="field-extra"><input class="field-description" value="${esc(schema.description || "")}" placeholder="Mô tả field"/><input class="field-default" value="${Object.hasOwn(schema, "default") ? esc(JSON.stringify(schema.default)) : ""}" placeholder="Default dạng JSON (tùy chọn)"/></div>
      ${typeConfigMarkup(type, schema)}
    </article>`;
  }

  function typeConfigMarkup(type, schema) {
    if (type === "object") {
      const props = schema.properties || {};
      const required = new Set(schema.required || []);
      const children = Object.entries(props).map(([name, child]) => schemaFieldMarkup(name, { ...child, __required: required.has(name) }, true)).join("");
      return `<div class="nested-fields"><div class="nested-label">Object properties</div><div class="field-list" data-field-list="nested">${children}</div><button class="add-field" data-add-field="nested">＋ Thêm property</button><label class="check-control"><input class="object-additional" type="checkbox" ${schema.additionalProperties !== false ? "checked" : ""}/> Cho phép property khác</label></div>`;
    }
    if (type === "array") {
      const item = schema.items && typeof schema.items === "object" ? schema.items : { type: "string" };
      return `<div class="array-items"><div class="nested-label">Array items</div>${schemaFieldMarkup("", item, false)}</div>`;
    }
    const constraints = schema.enum ? JSON.stringify(schema.enum) : "";
    const min = schema.minimum ?? schema.minLength ?? "";
    const max = schema.maximum ?? schema.maxLength ?? "";
    const pattern = schema.pattern || "";
    return `<div class="field-extra constraints"><input class="constraint-min" value="${esc(min)}" placeholder="Minimum / minLength"/><input class="constraint-max" value="${esc(max)}" placeholder="Maximum / maxLength"/><input class="constraint-enum" value="${esc(constraints)}" placeholder='Enum dạng JSON: ["a","b"]'/><input class="constraint-pattern" value="${esc(pattern)}" placeholder="Regex pattern (string)"/></div>`;
  }

  function bindSchemaEditor(root) {
    $$("[data-field-list='root'] > .schema-field", root).forEach(bindSchemaField);
    $$('[data-add-field="root"]', root).forEach(bindAddButton);
  }

  function bindSchemaField(field) {
    const typeSelect = $(".field-type", field);
    typeSelect.addEventListener("change", () => {
      const oldConfig = $(".field-extra:not(:first-child)", field);
      const oldChildren = $(".nested-fields, .array-items", field);
      if (oldConfig && oldConfig.classList.contains("constraints")) oldConfig.remove();
      oldChildren?.remove();
      const holder = document.createElement("div");
      holder.innerHTML = typeConfigMarkup(typeSelect.value, {});
      field.append(holder.firstElementChild);
      $$(".schema-field", field).forEach((child) => bindSchemaField(child));
      $$("[data-add-field]", field).forEach(bindAddButton);
    });
    const remove = $(".remove-field", field);
    remove.addEventListener("click", () => field.remove());
    $$("[data-add-field]", field).forEach(bindAddButton);
    $$(".schema-field", field).forEach((child) => {
      if (child !== field) bindSchemaField(child);
    });
  }

  function bindAddButton(button) {
    if (button.dataset.bound === "true") return;
    button.dataset.bound = "true";
    button.addEventListener("click", () => {
      const type = button.dataset.addField;
      const list = button.parentElement.querySelector(`[data-field-list="${type}"]`);
      if (!list) return;
      const shell = document.createElement("div");
      shell.innerHTML = schemaFieldMarkup("", { type: "string" }, true);
      const field = shell.firstElementChild;
      list.append(field);
      bindSchemaField(field);
    });
  }

  function collectSchemaField(field) {
    const type = $(".field-type", field).value;
    const schema = { type };
    const description = $(".field-description", field)?.value.trim();
    if (description) schema.description = description;
    const defaultText = $(".field-default", field)?.value.trim();
    if (defaultText) {
      try { schema.default = JSON.parse(defaultText); }
      catch { throw new Error(`Default của ${$(".field-name", field)?.value || "array item"} phải là JSON hợp lệ.`); }
    }
    const constraints = $(".constraints", field);
    if (constraints) {
      const min = $(".constraint-min", constraints).value.trim();
      const max = $(".constraint-max", constraints).value.trim();
      const pattern = $(".constraint-pattern", constraints).value.trim();
      const enumText = $(".constraint-enum", constraints).value.trim();
      if (min) schema[type === "string" ? "minLength" : "minimum"] = Number(min);
      if (max) schema[type === "string" ? "maxLength" : "maximum"] = Number(max);
      if (pattern) schema.pattern = pattern;
      if (enumText) {
        try { schema.enum = JSON.parse(enumText); }
        catch { throw new Error("Enum phải là JSON array hợp lệ."); }
        if (!Array.isArray(schema.enum)) throw new Error("Enum cần khai báo bằng JSON array.");
      }
    }
    if (type === "object") {
      const properties = {};
      const required = [];
      $$(":scope > .schema-field", $("[data-field-list='nested']", field)).forEach((child) => {
        const name = $(".field-name", child).value.trim();
        if (!name) throw new Error("Mỗi property cần có tên.");
        if (Object.hasOwn(properties, name)) throw new Error(`Property '${name}' bị trùng.`);
        properties[name] = collectSchemaField(child);
        if ($(".field-required", child).checked) required.push(name);
      });
      schema.properties = properties;
      schema.required = required;
      schema.additionalProperties = $(".object-additional", field).checked;
    }
    if (type === "array") {
      const itemField = $(".array-items > .schema-field", field);
      schema.items = itemField ? collectSchemaField(itemField) : { type: "string" };
    }
    return schema;
  }

  function collectTopSchema() {
    const properties = {};
    const required = [];
    $$("#wizard-content [data-field-list='root'] > .schema-field").forEach((field) => {
      const name = $(".field-name", field).value.trim();
      if (!name) throw new Error("Mỗi field cần có tên.");
      if (Object.hasOwn(properties, name)) throw new Error(`Field '${name}' bị trùng.`);
      properties[name] = collectSchemaField(field);
      if ($(".field-required", field).checked) required.push(name);
    });
    return { "$schema": "https://json-schema.org/draft/2020-12/schema", type: "object", properties, required, additionalProperties: $("#root-additional").checked };
  }

  function renderPolicyStep() {
    const policies = state.definition.field_policies || {};
    const fields = Object.entries(state.definition.input_schema.properties || {});
    if (!fields.length) return `<div class="empty-cell">Thêm ít nhất một field ở bước Cấu trúc dữ liệu để khai báo policies.</div>`;
    return `<div class="json-help">Model extraction chỉ chạy khi ứng dụng có provider được đăng ký. Nếu chưa cấu hình, chọn manual hoặc regex để tiếp tục.</div>
      <div class="policy-table"><div class="policy-row"><span>Field</span><span>Method</span><span>Required</span><span>Missing behavior</span></div>
      ${fields.map(([name, schema]) => policyRowMarkup(name, schema, policies[name] || {})).join("")}</div>`;
  }

  function policyRowMarkup(name, schema, policy) {
    const extraction = policy.extraction || { method: "manual" };
    const missing = policy.on_missing || { action: schema.default !== undefined ? "use_default" : "ask_user" };
    return `<div class="policy-row" data-policy="${esc(name)}"><strong>${esc(name)}</strong>
      <select class="policy-method">${EXTRACTION_METHODS.map((method) => `<option ${extraction.method === method ? "selected" : ""}>${method}</option>`).join("")}</select>
      <label class="check-control"><input class="policy-required" type="checkbox" ${policy.required ?? ((state.definition.input_schema.required || []).includes(name)) ? "checked" : ""}/> Required</label>
      <select class="policy-missing">${MISSING_ACTIONS.map((action) => `<option ${missing.action === action ? "selected" : ""}>${action}</option>`).join("")}</select>
      <div class="policy-pattern"><input class="policy-pattern-value" value="${esc(extraction.pattern || extraction.path || extraction.parser || "")}" placeholder="Regex pattern / parser name / path"/><label class="check-control"><input class="policy-multiple" type="checkbox" ${extraction.multiple ? "checked" : ""}/> Multiple values</label><span class="helper">${esc(schema.description || "")}</span></div>
    </div>`;
  }

  function bindPolicyEditor(root) {
    $$(".policy-row[data-policy]", root).forEach((row) => {
      $(".policy-method", row).addEventListener("change", () => {
        const method = $(".policy-method", row).value;
        $(".policy-pattern-value", row).placeholder = method === "regex" ? "Regex pattern" : method === "path" ? "JSON path" : method === "parser" ? "Registered parser name" : "Provider configuration (optional)";
      });
    });
  }

  function collectPolicies() {
    const result = {};
    $$(".policy-row[data-policy]").forEach((row) => {
      const name = row.dataset.policy;
      const method = $(".policy-method", row).value;
      const config = { method };
      const value = $(".policy-pattern-value", row).value.trim();
      if (value && method === "regex") { config.pattern = value; config.capture_group = 1; }
      if (value && method === "path") config.path = value;
      if (value && method === "parser") config.parser = value;
      if ($( ".policy-multiple", row).checked) config.multiple = true;
      const required = $(".policy-required", row).checked;
      const action = $(".policy-missing", row).value;
      const property = state.definition.input_schema.properties[name];
      const defaultValue = property.default;
      const onMissing = { action };
      if (action === "use_default" && defaultValue !== undefined) onMissing.value = defaultValue;
      if (action === "use_value" && defaultValue !== undefined) onMissing.value = defaultValue;
      result[name] = { value_type: property.type, extraction: config, required, on_missing: onMissing };
      if (property.description) result[name].description = property.description;
    });
    return result;
  }

  function renderTransformStep() {
    return `<div class="json-help">JSONata nhận input đã validate và tạo JSON mới cho renderer. Identity là <code>$</code>. Expression được thực thi trong runtime có guardrails.</div>
      <div class="form-control"><label for="transform-expression">JSONata expression</label><textarea id="transform-expression" style="min-height:150px">${esc(state.definition.transform.expression || "$")}</textarea><span class="helper">Ví dụ: <code>{"name": student.name, "scores": student.scores}</code></span></div>`;
  }

  function renderRenderStep() {
    const templates = state.definition.render.templates || { latex: "" };
    return `<div class="json-help">Jinja2 chạy trong sandbox. Với dữ liệu văn bản LaTeX, dùng filter <code>latex_escape</code>; LaTeX sẽ được static-check trước khi trả về.</div>
      <div id="template-list">${Object.entries(templates).map(([target, template]) => templateCard(target, template)).join("")}</div>
      <button class="add-field" id="add-template">＋ Thêm output format</button>`;
  }

  function templateCard(target, template) {
    return `<article class="target-card"><div class="target-head"><input class="template-target" value="${esc(target)}" placeholder="latex / mathjax / obsidian"/><button class="remove-field remove-template" type="button" aria-label="Xóa template">×</button></div><textarea class="template-source" placeholder="Jinja2 template">${esc(template)}</textarea></article>`;
  }

  function bindTemplateEditor(root) {
    $("#add-template", root).addEventListener("click", () => {
      const host = $("#template-list", root);
      const shell = document.createElement("div");
      shell.innerHTML = templateCard("", "");
      host.append(shell.firstElementChild);
      bindTemplateRemove(host.lastElementChild);
    });
    $$(".remove-template", root).forEach(bindTemplateRemove);
  }

  function bindTemplateRemove(button) {
    button.addEventListener("click", () => button.closest(".target-card").remove());
  }

  function renderPublishStep() {
    const definition = buildDefinition(false);
    const json = JSON.stringify(definition, null, 2);
    return `<div class="json-help">Validate và preview với dữ liệu mẫu trước khi publish. Intent trùng cần xác nhận ghi đè; sau publish UI sẽ reload registry.</div>
      <div class="form-control"><label for="preview-values">Dữ liệu preview (JSON object)</label><textarea id="preview-values" class="code-input">{}</textarea></div>
      <div class="step-nav"><button class="step-jump" id="validate-now">Validate definition</button><button class="step-jump" id="preview-now">Preview output</button><button class="step-jump" id="publish-now">${state.editing ? "Publish update" : "Publish definition"}</button></div>
      <div id="validation-result" class="helper" style="margin-top:12px"></div><pre id="preview-result" class="execute-result">Chưa chạy preview.</pre>
      <details class="definition-details"><summary>Definition JSON</summary><textarea id="definition-preview" readonly class="code-input">${esc(json)}</textarea></details>`;
  }

  function buildDefinition(validateSchema = true) {
    if (validateSchema && state.step === 1) state.definition.input_schema = collectTopSchema();
    const definition = structuredClone(state.definition);
    definition.definition_version = "2.0";
    definition.transform = { engine: "jsonata", expression: $("#transform-expression")?.value ?? definition.transform.expression ?? "$", mode: "expression" };
    if (definition.input_schema.type !== "object") throw new Error("input_schema root phải là object.");
    return definition;
  }

  function captureStep() {
    if (state.step === 0) {
      const get = (key) => $(`[data-general="${key}"]`).value.trim();
      state.definition.intent = get("intent");
      state.definition.category = get("category");
      state.definition.domain = get("domain");
      state.definition.describe = get("describe");
      state.definition.aliases = get("aliases").split("\n").map((item) => item.trim()).filter(Boolean);
      if (!state.definition.intent) throw new Error("Intent ID là bắt buộc.");
      if (!state.definition.category || !state.definition.domain) throw new Error("Category và domain là bắt buộc.");
    } else if (state.step === 1) {
      state.definition.input_schema = collectTopSchema();
      state.definition.field_policies = Object.fromEntries(
        Object.entries(state.definition.field_policies || {}).filter(([key]) => Object.hasOwn(state.definition.input_schema.properties, key)),
      );
    } else if (state.step === 2) {
      state.definition.field_policies = collectPolicies();
    } else if (state.step === 3) {
      state.definition.transform.expression = $("#transform-expression").value.trim();
      if (!state.definition.transform.expression) throw new Error("JSONata expression không được rỗng.");
    } else if (state.step === 4) {
      const templates = {};
      $$(".target-card").forEach((card) => {
        const target = $(".template-target", card).value.trim();
        if (!target) throw new Error("Mỗi output format cần có tên.");
        if (Object.hasOwn(templates, target)) throw new Error(`Output format '${target}' bị trùng.`);
        templates[target] = $(".template-source", card).value;
      });
      if (!Object.keys(templates).length) throw new Error("Cần ít nhất một render template.");
      state.definition.render = { engine: "jinja2", context: "normalized_input", templates };
      state.definition.templates = { latex: templates.latex || "", python: "", cpp: "" };
    }
  }

  function goToStep(nextStep) {
    try {
      if (nextStep > state.step) captureStep();
      state.step = Math.max(0, Math.min(WIZARD_STEPS.length - 1, nextStep));
      renderWizard();
    } catch (error) {
      toast(error.message, "error");
    }
  }

  async function validateCurrentDefinition() {
    try {
      captureStep();
      const definition = buildDefinition(false);
      const result = await api("/validate", { method: "POST", body: JSON.stringify({ definition }) });
      state.previewedDefinition = "";
      const holder = $("#validation-result");
      holder.textContent = result.valid ? `✓ Contract hợp lệ · ${definition.intent}` : result.errors.map((item) => `${item.path}: ${item.message}`).join("\n");
      holder.className = result.valid ? "inline-message success-text" : "inline-message error-text";
      if (result.valid) toast("Definition vượt qua validation.", "success");
    } catch (error) {
      toast(error.message, "error");
    }
  }

  async function previewCurrentDefinition() {
    try {
      const definition = buildDefinition(false);
      const validation = await api("/validate", { method: "POST", body: JSON.stringify({ definition }) });
      if (!validation.valid) throw new Error(validation.errors.map((item) => `${item.path}: ${item.message}`).join("\n"));
      const values = JSON.parse($("#preview-values").value || "{}");
      if (!values || typeof values !== "object" || Array.isArray(values)) throw new Error("Preview values cần là JSON object.");
      const result = await api("/preview", { method: "POST", body: JSON.stringify({ definition, values, target: "latex" }) });
      $("#preview-result").textContent = JSON.stringify(result, null, 2);
      state.previewedDefinition = result.status === "ok" ? JSON.stringify(definition) : "";
      $("#validation-result").textContent = result.status === "ok" ? "Preview thành công; có thể publish." : `Preview chưa hoàn tất: ${result.missing?.join(", ") || result.status}`;
      $("#validation-result").className = result.status === "ok" ? "inline-message success-text" : "inline-message error-text";
    } catch (error) {
      $("#validation-result").textContent = error.message;
      $("#validation-result").className = "inline-message error-text";
      toast(error.message, "error");
    }
  }

  async function publishCurrentDefinition() {
    try {
      const definition = buildDefinition(false);
      const validation = await api("/validate", { method: "POST", body: JSON.stringify({ definition }) });
      if (!validation.valid) throw new Error(validation.errors.map((item) => `${item.path}: ${item.message}`).join("\n"));
      if (state.previewedDefinition !== JSON.stringify(definition)) throw new Error("Hãy chạy preview thành công với definition hiện tại trước khi publish.");
      const exists = state.definitions.some((item) => item.intent === definition.intent);
      const overwrite = exists && window.confirm(`Intent ${definition.intent} đã tồn tại. Bạn xác nhận ghi đè definition này chứ?`);
      if (exists && !overwrite) return;
      const result = await api("/publish", { method: "POST", body: JSON.stringify({ definition, overwrite }) });
      await api("/reload", { method: "POST" });
      toast(`Đã publish và reload ${result.intent}.`, "success");
      state.editing = true;
      state.definition = definition;
      setView("dashboard");
    } catch (error) {
      toast(error.message, "error");
    }
  }

  async function exportDefinitions() {
    const result = await api("/export");
    const blob = new Blob([`${JSON.stringify(result.definitions, null, 2)}\n`], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "knowledge-definitions.json";
    link.click();
    URL.revokeObjectURL(link.href);
    toast(`Đã export ${result.count} definitions.`, "success");
  }

  async function importDefinitions() {
    const status = $("#import-status");
    try {
      const parsed = JSON.parse($("#import-json").value);
      const definitions = Array.isArray(parsed) ? parsed : parsed.definitions;
      if (!Array.isArray(definitions) || !definitions.length) throw new Error("JSON cần là array definition không rỗng hoặc object có key definitions.");
      const validation = await api("/import/validate", { method: "POST", body: JSON.stringify({ definitions }) });
      if (!validation.valid) throw new Error(validation.errors.map((item) => `${item.path}: ${item.message}`).join("\n"));
      const conflicts = validation.conflicts || [];
      const prompt = `Batch hợp lệ gồm ${definitions.length} definitions.${conflicts.length ? ` Sẽ ghi đè ${conflicts.join(", ")}.` : ""} Tiếp tục publish?`;
      if (!window.confirm(prompt)) return;
      let published = 0;
      try {
        for (const definition of definitions) {
          await api("/publish", { method: "POST", body: JSON.stringify({ definition, overwrite: conflicts.includes(definition.intent) }) });
          published += 1;
        }
        await api("/reload", { method: "POST" });
      } catch (error) {
        throw new Error(`Đã publish ${published}/${definitions.length}; batch dừng do: ${error.message}. Các definition đã publish vẫn được giữ.`);
      }
      status.textContent = `Đã import và reload ${published} definitions.`;
      status.className = "inline-message success-text";
      await loadDashboard();
      toast(status.textContent, "success");
    } catch (error) {
      status.textContent = error.message;
      status.className = "inline-message error-text";
    }
  }

  async function reloadDefinitions() {
    try {
      const result = await api("/reload", { method: "POST" });
      await loadDashboard();
      toast(`Registry đã reload (${result.version}).`, "success");
    } catch (error) { toast(error.message, "error"); }
  }

  async function rollbackDefinitions() {
    if (!window.confirm("Phục hồi toàn bộ definition bank về phiên bản trước? Thao tác này thay đổi dữ liệu đang dùng.")) return;
    try {
      const result = await api("/rollback", { method: "POST", body: JSON.stringify({}) });
      await api("/reload", { method: "POST" });
      await loadDashboard();
      toast(`Đã rollback registry (${result.version}).`, "success");
    } catch (error) { toast(error.message, "error"); }
  }

  async function executeQuery() {
    const query = $("#execute-query").value.trim();
    if (!query) { toast("Nhập câu cần thử.", "error"); return; }
    try {
      const result = await request("/ask", { method: "POST", body: JSON.stringify({ text: query, target: "latex" }) });
      $("#execute-result").textContent = JSON.stringify(result, null, 2);
    } catch (error) { $("#execute-result").textContent = error.message; }
  }

  function bindEvents() {
    $$(".tab").forEach((tab) => tab.addEventListener("click", () => setView(tab.dataset.view)));
    $("#new-definition").addEventListener("click", startNewDefinition);
    $("#new-definition-heading").addEventListener("click", startNewDefinition);
    $("#back-dashboard").addEventListener("click", () => setView("dashboard"));
    $("#refresh-definitions").addEventListener("click", loadDashboard);
    $("#export-definitions").addEventListener("click", () => exportDefinitions().catch((error) => toast(error.message, "error")));
    $("#open-import").addEventListener("click", () => $("#import-dialog").showModal());
    $("#cancel-import").addEventListener("click", () => $("#import-dialog").close());
    $("#validate-import").addEventListener("click", importDefinitions);
    $("#reload-definitions").addEventListener("click", reloadDefinitions);
    $("#rollback-definitions").addEventListener("click", rollbackDefinitions);
    $("#execute-query-button").addEventListener("click", executeQuery);
    $("#search-input").addEventListener("input", renderDefinitions);
    $("#category-filter").addEventListener("change", renderDefinitions);
    $("#domain-filter").addEventListener("change", renderDefinitions);
    $("#previous-step").addEventListener("click", () => goToStep(state.step - 1));
    $("#next-step").addEventListener("click", () => {
      if (state.step === WIZARD_STEPS.length - 1) {
        validateCurrentDefinition();
        return;
      }
      goToStep(state.step + 1);
    });
    $("#wizard-content").addEventListener("click", (event) => {
      if (event.target.id === "validate-now") validateCurrentDefinition();
      if (event.target.id === "preview-now") previewCurrentDefinition();
      if (event.target.id === "publish-now") publishCurrentDefinition();
    });
  }

  window.KAGui = { api, buildDefinition, collectSchemaField, freshDefinition, renderDefinitions, state };
  bindEvents();
  loadDashboard();
})();
