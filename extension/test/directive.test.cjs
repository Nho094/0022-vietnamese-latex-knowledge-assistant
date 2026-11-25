const test = require("node:test");
const assert = require("node:assert/strict");
const { parseKnowledgeDirective } = require("../dist/editor/directive.js");

test("parses line comments for supported languages", () => {
  const examples = [
    ["python", "# A: Tạo phân số", "Tạo phân số"],
    ["shellscript", "  # A: Tạo bảng", "Tạo bảng"],
    ["lua", "-- A: Tạo phương trình", "Tạo phương trình"],
    ["typescript", "// A: Tạo ma trận", "Tạo ma trận"],
    ["latex", "% A: Tạo căn bậc hai", "Tạo căn bậc hai"],
    ["css", "/* A: Tạo màu nổi bật */", "Tạo màu nổi bật"],
    ["markdown", "<!-- A: Tạo bảng -->", "Tạo bảng"],
  ];
  for (const [language, line, expected] of examples) {
    assert.equal(parseKnowledgeDirective(language, line), expected);
  }
});

test("ignores ordinary comments and unknown languages", () => {
  assert.equal(parseKnowledgeDirective("python", "# ordinary comment"), null);
  assert.equal(parseKnowledgeDirective("plaintext", "# A: Tạo bảng"), null);
});
