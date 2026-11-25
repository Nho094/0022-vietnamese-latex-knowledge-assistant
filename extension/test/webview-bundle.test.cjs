const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const dist = path.resolve(__dirname, "../webview/dist");

test("production Webview references local JavaScript and CSS assets", () => {
  const html = fs.readFileSync(path.join(dist, "index.html"), "utf8");
  assert.match(html, /\.\/assets\/app\.js/);
  assert.match(html, /\.\/assets\/app\.css/);
  assert.doesNotMatch(html, /(?:src|href)=["']https?:\/\//i);
  assert.ok(fs.statSync(path.join(dist, "assets", "app.js")).size > 50_000);
  assert.ok(fs.statSync(path.join(dist, "assets", "app.css")).size > 10_000);
});

test("production bundle contains real app screens and dark theme tokens", () => {
  const script = fs.readFileSync(path.join(dist, "assets", "app.js"), "utf8");
  const stylesheet = fs.readFileSync(path.join(dist, "assets", "app.css"), "utf8");
  assert.match(script, /Kiến thức vừa cập nhật/);
  assert.match(script, /Hỏi bằng ngôn ngữ tự nhiên/);
  assert.match(script, /Lịch sử hoạt động/);
  assert.match(stylesheet, /--background:#0b0d12/);
  assert.match(stylesheet, /\.runtime-overview/);
});
