const test = require("node:test");
const assert = require("node:assert/strict");
const {
  parseKnowledgeDirective,
  parseScopedDirective,
} = require("../dist/editor/directive.js");

test("parses line and block comments across universal languages", () => {
  const examples = [
    ["python", "# A: Tạo phân số", "Tạo phân số"],
    ["plaintext", "# A: Cách setup Sunshine trên Linux", "Cách setup Sunshine trên Linux"],
    ["shellscript", "  # A: Tạo bảng", "Tạo bảng"],
    ["rust", "// A: Viết hàm đọc file", "Viết hàm đọc file"],
    ["java", "// A: Tạo class Student", "Tạo class Student"],
    ["go", "// A: Tạo http server", "Tạo http server"],
    ["sql", "-- A: Truy vấn top 10", "Truy vấn top 10"],
    ["lua", "-- A: Tạo phương trình", "Tạo phương trình"],
    ["typescript", "// A: Tạo ma trận", "Tạo ma trận"],
    ["latex", "% A: Tạo căn bậc hai", "Tạo căn bậc hai"],
    ["tex", "% A: Viết tích phân", "Viết tích phân"],
    ["css", "/* A: Tạo màu nổi bật */", "Tạo màu nổi bật"],
    ["markdown", "<!-- A: Tạo bảng -->", "Tạo bảng"],
    ["markdown", "%% A: Tạo callout Obsidian %%", "Tạo callout Obsidian"],
    ["html", "<!-- A: Tạo thẻ card -->", "Tạo thẻ card"],
  ];
  for (const [language, line, expected] of examples) {
    assert.equal(parseKnowledgeDirective(language, line), expected);
  }
});

test("supports custom commentPrefixes from user configuration", () => {
  const custom = { prisma: ["//"], "*": ["#"] };
  assert.equal(parseKnowledgeDirective("prisma", "// A: Tạo bảng user", custom), "Tạo bảng user");
  assert.equal(parseKnowledgeDirective("customlang", "# A: Ghi chú nhanh", custom), "Ghi chú nhanh");
});

test("parses optional scope prefix without breaking colons inside queries", () => {
  const scoped = parseScopedDirective("plaintext", "# A: py: viết hàm fibonacci");
  assert.deepEqual(scoped, {
    rawQuery: "py: viết hàm fibonacci",
    query: "viết hàm fibonacci",
    scope: "py",
    targetOverride: "python",
    categoryOverride: "python",
  });

  const sysadmin = parseScopedDirective("plaintext", "# A: sunshine: mở port 47989");
  assert.deepEqual(sysadmin, {
    rawQuery: "sunshine: mở port 47989",
    query: "mở port 47989",
    scope: "sunshine",
    targetOverride: "text",
    categoryOverride: "sysadmin",
  });

  const nonScopeColon = parseScopedDirective("plaintext", "# A: Cài máy in: chia sẻ SMB");
  assert.deepEqual(nonScopeColon, {
    rawQuery: "Cài máy in: chia sẻ SMB",
    query: "Cài máy in: chia sẻ SMB",
  });
});

test("ignores ordinary comments and unknown languages without fallback", () => {
  assert.equal(parseKnowledgeDirective("python", "# ordinary comment"), null);
  assert.equal(parseKnowledgeDirective("unknown_lang_xyz", "# A: Tạo bảng"), null);
});
