"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.directiveCommentPrefixes = void 0;
exports.parseKnowledgeDirective = parseKnowledgeDirective;
exports.directiveCommentPrefixes = {
    python: ["#"],
    shellscript: ["#"],
    lua: ["--"],
    javascript: ["//"],
    javascriptreact: ["//"],
    typescript: ["//"],
    typescriptreact: ["//"],
    c: ["//"],
    cpp: ["//"],
    css: ["/*"],
    scss: ["/*"],
    latex: ["%"],
    markdown: ["<!--"],
};
/** Lấy yêu cầu khỏi comment theo ngôn ngữ; trả null khi dòng không phải chỉ thị. */
function parseKnowledgeDirective(languageId, line) {
    const prefixes = exports.directiveCommentPrefixes[languageId];
    if (!prefixes)
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
        const match = /^A\s*:\s*(.+)$/i.exec(body);
        return match?.[1]?.trim() || null;
    }
    return null;
}
//# sourceMappingURL=directive.js.map