"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.directiveCommentPrefixes = void 0;
exports.resolveDirectivePrefixes = resolveDirectivePrefixes;
exports.parseKnowledgeDirective = parseKnowledgeDirective;
exports.parseScopedDirective = parseScopedDirective;
exports.directiveCommentPrefixes = {
    python: ["#"],
    shellscript: ["#"],
    plaintext: ["#"],
    yaml: ["#"],
    toml: ["#"],
    ini: ["#"],
    powershell: ["#"],
    dockerfile: ["#"],
    makefile: ["#"],
    r: ["#"],
    ruby: ["#"],
    perl: ["#"],
    lua: ["--"],
    sql: ["--"],
    haskell: ["--"],
    javascript: ["//"],
    javascriptreact: ["//"],
    typescript: ["//"],
    typescriptreact: ["//"],
    c: ["//"],
    cpp: ["//"],
    csharp: ["//"],
    rust: ["//"],
    java: ["//"],
    go: ["//"],
    kotlin: ["//"],
    swift: ["//"],
    dart: ["//"],
    jsonc: ["//"],
    php: ["//", "#"],
    css: ["/*"],
    scss: ["/*"],
    less: ["/*"],
    latex: ["%"],
    tex: ["%"],
    plaintex: ["%"],
    bibtex: ["%"],
    matlab: ["%"],
    markdown: ["<!--", "%%"],
    mdx: ["<!--", "%%"],
    html: ["<!--"],
    xml: ["<!--"],
    svg: ["<!--"],
};
const SCOPE_ROUTING_MAP = {
    py: { targetOverride: "python", categoryOverride: "python" },
    python: { targetOverride: "python", categoryOverride: "python" },
    rs: { targetOverride: "rust", categoryOverride: "rust" },
    rust: { targetOverride: "rust", categoryOverride: "rust" },
    cpp: { targetOverride: "cpp", categoryOverride: "cpp" },
    "c++": { targetOverride: "cpp", categoryOverride: "cpp" },
    java: { targetOverride: "java", categoryOverride: "java" },
    js: { targetOverride: "javascript", categoryOverride: "javascript" },
    javascript: { targetOverride: "javascript", categoryOverride: "javascript" },
    ts: { targetOverride: "typescript", categoryOverride: "javascript" },
    typescript: { targetOverride: "typescript", categoryOverride: "javascript" },
    css: { targetOverride: "css", categoryOverride: "css" },
    html: { targetOverride: "html", categoryOverride: "css" },
    sql: { targetOverride: "sql", categoryOverride: "shell" },
    lua: { targetOverride: "lua", categoryOverride: "shell" },
    sh: { targetOverride: "shell", categoryOverride: "shell" },
    shell: { targetOverride: "shell", categoryOverride: "shell" },
    bash: { targetOverride: "shell", categoryOverride: "shell" },
    linux: { targetOverride: "shell", categoryOverride: "shell" },
    sys: { targetOverride: "text", categoryOverride: "sysadmin" },
    sysadmin: { targetOverride: "text", categoryOverride: "sysadmin" },
    sunshine: { targetOverride: "text", categoryOverride: "sysadmin" },
    word: { targetOverride: "text", categoryOverride: "office" },
    office: { targetOverride: "text", categoryOverride: "office" },
    excel: { targetOverride: "text", categoryOverride: "office" },
    txt: { targetOverride: "auto" },
    text: { targetOverride: "auto" },
    general: { targetOverride: "auto", categoryOverride: "general" },
    all: { targetOverride: "auto" },
    tex: { targetOverride: "latex" },
    latex: { targetOverride: "latex" },
    mjx: { targetOverride: "mathjax" },
    mathjax: { targetOverride: "mathjax" },
    md: { targetOverride: "markdown" },
    markdown: { targetOverride: "markdown" },
    obs: { targetOverride: "obsidian" },
    obsidian: { targetOverride: "obsidian" },
};
function normalizePrefixList(value) {
    if (!value)
        return [];
    const items = typeof value === "string" ? [value] : value;
    return items.map((item) => item.trim()).filter((item) => item.length > 0);
}
/** Lấy danh sách tiền tố comment hiệu lực cho languageId (ưu tiên cấu hình người dùng). */
function resolveDirectivePrefixes(languageId, customPrefixes = {}) {
    const exactCustom = normalizePrefixList(customPrefixes[languageId]);
    const wildcardCustom = normalizePrefixList(customPrefixes["*"]);
    const builtIn = exports.directiveCommentPrefixes[languageId] ?? [];
    const merged = [...exactCustom, ...builtIn, ...wildcardCustom];
    return Array.from(new Set(merged));
}
/** Lấy yêu cầu khỏi comment theo ngôn ngữ; trả null khi dòng không phải chỉ thị. */
function parseKnowledgeDirective(languageId, line, customPrefixes = {}) {
    const prefixes = resolveDirectivePrefixes(languageId, customPrefixes);
    if (prefixes.length === 0)
        return null;
    const trimmed = line.trim();
    for (const prefix of prefixes) {
        if (!trimmed.startsWith(prefix))
            continue;
        let body = trimmed.slice(prefix.length).trimStart();
        if (prefix === "/*")
            body = body.replace(/\*\/$/, "").trimEnd();
        if (prefix === "<!--")
            body = body.replace(/-->$/, "").trimEnd();
        if (prefix === "%%")
            body = body.replace(/%%$/, "").trimEnd();
        const match = /^A\s*:\s*(.+)$/i.exec(body);
        if (match?.[1]?.trim()) {
            return match[1].trim();
        }
    }
    return null;
}
/** Tách scope tùy chọn (`# A: py: viết hàm fib`) từ chỉ thị comment để định tuyến tầng 2. */
function parseScopedDirective(languageId, line, customPrefixes = {}) {
    const rawQuery = parseKnowledgeDirective(languageId, line, customPrefixes);
    if (!rawQuery)
        return null;
    const scopeMatch = /^([a-z0-9_+-]{2,16})\s*:\s*(.+)$/i.exec(rawQuery);
    if (scopeMatch) {
        const scope = scopeMatch[1].toLowerCase();
        const mapped = SCOPE_ROUTING_MAP[scope];
        if (mapped && scopeMatch[2].trim()) {
            return {
                rawQuery,
                query: scopeMatch[2].trim(),
                scope,
                targetOverride: mapped.targetOverride,
                categoryOverride: mapped.categoryOverride,
            };
        }
    }
    return { rawQuery, query: rawQuery };
}
//# sourceMappingURL=directive.js.map