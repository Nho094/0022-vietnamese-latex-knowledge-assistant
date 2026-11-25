# VS Code Extension Implementation Plan

## Current architecture

The project already has a Python FastAPI service in `api/server.py`, a generic Contract v2 engine under `core/definitions`, `core/engine`, `core/extraction`, `core/policies`, `core/processing`, and `core/rendering`, plus an atomic definition registry with snapshots, backups, publish, reload, and rollback. Existing management routes cover list/detail, validate, preview, publish, import validation/export, reload, rollback, corpus listing, and `/ask`. The current `gui/static` page is a browser management UI, not a VS Code extension. It exposes backend contract terms and lacks Editor integration, lifecycle management, soft status, and durable user-facing activity history.

## Reuse from the dashboard repository

`shadcndashboard` is an already-cloned Vite 8 + React 19 + TypeScript + Tailwind v4 single-page app. Its source is a client-only dashboard with mock data and is not a backend to reuse. Its UI primitives use Base UI and Tailwind utility classes. We will selectively adapt the compact sidebar/page shell, table and form layouts, dark/light theme tokens, and keyboard/focus patterns; we will not copy the mock dashboard routes or chart-heavy pages. The source repository is MIT licensed (copyright Shadcn Dashboard, 2026); the extension will retain its MIT notice and add an attribution file. The repository itself remains unchanged.

## Backend reuse and additions

Keep FastAPI, the NLU pipeline, Contract v2 loader, generic engine, registry, retrieval, policy, transform, and sandboxed renderers as the only knowledge-processing stack. Extend management APIs only for missing operations: local service status, safe deletion with the existing registry backup/rollback path, and history listing. The current project has no configured model suitable for generating knowledge definitions, so this version does not claim to provide AI-assisted suggestions; the no-code Builder works in manual mode and reports the limitation.

Extension-owned activity, enabled/disabled state, and the active definition bank live in VS Code `globalStorage`. On first launch only, the bundled `intents.json` seeds that storage using exclusive create; later starts never replace a user's bank. Corpus data stays read-only in the extension bundle. Definition writes continue through validate → preview → explicit publish, with overwrite confirmation and registry backups. Import validates the full batch before any publish. Deletion is an explicit confirmed operation and remains recoverable through registry snapshots.

## Extension and Webview architecture

- Root VS Code manifest and TypeScript Extension Host own commands, Code Actions, backend process lifecycle, Webview CSP, local API calls, session-token injection, activity storage, and editor insertion.
- `extension/webview` is a Vite + React + TypeScript + Tailwind Webview bundle. It communicates only with the Extension Host using typed `postMessage` messages; it never calls localhost or Python directly.
- The Extension Host starts the packaged FastAPI source on a loopback-only ephemeral port, passes a random per-session token, polls health, reports missing Python/dependencies/model as degraded states, and terminates only the child it started.
- The Python service honors the chosen loopback port and validates the session token when one is configured. The health probe is narrowly exempt so startup can be observed.
- Editor directives are parsed by language-specific comment delimiters from a shared declarative map. A Code Action and Command Palette command preview the generated result; insertion happens only after user confirmation.

## Data preservation and compatibility

No existing definition, test set, legacy directory, or corpus is moved or rewritten. New extension state uses VS Code `globalStorageUri`. Existing API behavior and schemas remain compatible; new routes are additive and session auth is opt-in for extension-managed service instances. Python dependencies and model weights are not copied into the VSIX. The Extension checks the selected interpreter and reports setup/model limitations instead of claiming full operation or downloading anything silently.

## Phases

1. Audit repository and dashboard source; this plan records reusable components, API gaps, license, and storage constraints.
2. Create a minimal Webview build and shared design tokens; selectively adapt the dashboard shell, form, and table patterns; implement dark/light themes and accessible shared controls.
3. Implement the Extension Host, secure loopback backend process manager, typed message bridge, commands, Code Actions, and explicit preview/insert flow.
4. Build connected Dashboard, Explorer, four-step Builder, Detail, History, and Settings. Keep business logic in FastAPI and state in the correct backend/extension store.
5. Add API routes only where the audited service is missing functionality; retain generic Contract v2 processing, registry backup, validation, preview, publish, and rollback.
6. Add unit/integration tests for API security, CRUD/history, Builder policies, Webview bridge, directives, missing model, process lifecycle, and regression behavior.
7. Build the Webview and extension, package a VSIX if `vsce` is available, inspect its contents, and document environment limitations honestly.

## Verification

Use the existing Python suite and Ruff plus Webview type-check/build/lint and extension tests. Test the VSIX manifest and packaged files. Do not run or alter `data/test.csv`; preserve its existing SHA-256. Browser automation and real model integration are reported as blocked if their tools/checkpoints are unavailable.

## UI redesign and packaging result

The Webview now uses a compact developer-tool shell with a default dark theme and the existing light theme, grouped navigation, connected service/model status, a real Quick Ask workspace, filtered/paginated knowledge browsing, and clearer Settings status. Existing API messages and backend contracts remain unchanged. Dashboard activity uses only activity supplied by the Extension Host and shows an empty state when none exists. Production Vite assets are fixed to `assets/app.js` and `assets/app.css`, matching the Webview resource loader.

`npm run typecheck`, `npm run build`, `npm run test:extension` (2 passed), and `npm run package` passed. The resulting VSIX is `dist/vietnamese-knowledge-assistant-0.1.0.vsix` (60 files, about 359 KB). Its manifest, backend source, runtime JSON data, Extension Host, and local Webview assets were inspected; training/test CSVs and model caches are excluded. Installing into a temporary VS Code profile succeeded. Installing into the current VS Code profile was blocked because the sandbox makes `~/.config/Code` read-only; actual interactive Webview/backend operation and browser automation are therefore not claimed as verified. Python, Python packages, and model checkpoints are still external prerequisites; the VSIX is not standalone.

## Runtime model discovery follow-up

The Extension now selects `vietnameseKnowledgeAssistant.modelRoot`, then `<workspace>/models`, or `globalStorage/models` when no workspace is open. It passes explicit checkpoint, Hugging Face cache, SentenceTransformer, and index-cache paths to the backend and keeps Hugging Face in offline mode. Python also searches standard Hugging Face cache locations when workspace artifacts are absent. Config/status distinguish a fine-tuned PhoBERT classifier, retrieval model, index cache, and missing Python packages. Index cache validation uses the same model ID, registry-derived hash, preprocessing fingerprint, row count, and array dimensions as the retrieval loader. A missing model does not prevent the management API from starting; `/ask` returns actionable `unknown` when no NLU component can run. The classifier loader accepts both `.bin` and `.safetensors` files.

Verification: 578 passed, 1 skipped, 7 warnings; Ruff, validator, test-set hash check, TypeScript, build, 3 Extension tests, package, and temporary-profile VSIX installation passed. The audit environment found retrieval weights in the standard Hugging Face cache and successfully ran an offline retrieval-only CLI query. PhoBERT checkpoint is absent, so classifier runtime was not verified. HTTP loopback and interactive VS Code smoke remain blocked/unverified in the sandbox.
