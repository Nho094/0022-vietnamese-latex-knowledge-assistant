# KA-15: Generic Knowledge Engine architecture audit

**Status:** audit/design complete; implementation has not started. This milestone is read-only with respect to application code, intent data, training data, and knowledge corpora.

## Current system and contract

The active runtime remains the legacy scenario pipeline. `Pipeline._load_intents_bank` loads a list into an in-memory map (`core/pipeline.py`); extraction reads `slots` (`core/slots.py`); rendering reads `templates`, `kind`, `requires`, and `post` (`core/pipeline.py`, `core/latex_tools.py`). `core.validator` validates this legacy contract. A source scan of `core/`, `api/`, and `cli/` found no runtime references to `definition_version`, `input_schema`, `field_policies`, `transform`, or the v2 `render` field.

The Data Contract v2 artifacts do exist: `data/standards/manifest.json`, the Draft 2020-12 definition schema, four corpus schemas, a catalog, an example with nested arrays, and a migration report. The current bank contains 189 definitions; each has `definition_version`, `input_schema`, `field_policies`, `transform`, and `render`, while retaining the legacy fields. The new fields are data artifacts today, not an executable engine.

Observed coverage in the current definitions:

- 98 definitions have slots, represented by 174 legacy scalar slots and 174 regex policies. The bank has no model/parser/path/manual policy examples; those methods are only enumerated by the contract schema.
- 188 definitions have v2 render templates; one is the intentionally empty `OUT_OF_SCOPE`. Current v2 output keys are 105 MathJax, 42 LaTeX, and 41 Obsidian templates.
- 146 definitions omit `domain`; only 43 declare it. Domain/category fallback is currently implicit in legacy behavior and must be made explicit in the compatibility rules.
- All current transform expressions use identity (`$`). The nested dynamic-table example demonstrates data shape, but does not demonstrate a working extractor, JSONata processor, or renderer runtime.

## Verified architecture boundaries

| Area | Current implementation | Compatibility consequence |
|---|---|---|
| PhoBERT | Fixed classifier label map and checkpoint/fingerprint checks (`core/classifier.py`) | New definitions must not be presented as PhoBERT classes until a compatible classifier is trained. Route new/unknown definitions through the live retrieval registry. |
| SBERT retrieval | Loads aliases/scenario representations; supports in-process `add_intent` (`core/retrieval.py`) | Reuse encoder/indexing and hash logic; coordinate durable definition writes, index snapshots, cache invalidation, rollback, and concurrent readers. |
| Fuzzy/TF-IDF | Existing alternatives and ablation paths | Keep behind their existing interfaces; do not fold them into schema processing or remove them. |
| Extraction | Regex over scalar legacy slots (`core/slots.py`) | Add policy adapters behind a generic interface; keep a legacy adapter and leave model extraction unavailable unless a provider is configured. |
| Validation | Project-specific legacy intent checks (`core/validator.py`) | Add full recursive Draft 2020-12 validation without removing legacy lint or current error contracts. |
| Transform/render | Legacy LaTeX substitution and static security check (`core/latex_tools.py`) | Add JSONata and sandboxed Jinja2 as separate processors; LaTeX escaping and static checks remain mandatory for LaTeX output. |
| API/CLI | `POST /ask`, `/healthz`, CLI `kb add/list/check`; API owns a process-global pipeline | Preserve old routes/payloads. Add management/version endpoints and a safe registry reload path rather than moving business rules into the GUI. |
| GUI | No GUI source directory or workflow tests found | Build a local client over management APIs after the backend contract is stable. |
| Corpus | Four corpora are cataloged separately from intent definitions | Preserve their schemas, paths, and content; corpus management is a separate feature from definition management. |

`jsonschema` and `jsonata` are absent from the active environment and are not direct requirements. Jinja2 is present in the lock/environment transitively, but is not a direct requirement. No runtime `SandboxedEnvironment`, JSONata execution, generic registry reload, or v2 contract tests were found. The intent definition schema's `$id` uses `example.invalid`; `input_schema` is currently described only as an object shell, so nested Draft 2020-12 validity is not established by the active validator.

The migration report says its source archive was `data(1).zip`, but that archive is absent from this checkout. Its preservation claims and Jinja compilation results are recorded artifacts, not independently reproducible evidence in this session. Do not rewrite the report or infer lost source values; obtain the archive or an owner-approved baseline before any migration that needs byte-for-byte source comparison.

## Data preservation baseline

No project data was modified in KA-15. Hashes captured for future migrations:

| File | SHA-256 |
|---|---|
| `data/intents.json` | `3376514571bc4ba516448d7395e444b41705f655d772ced756fd3be39e347478` |
| `data/test.csv` | `ef8781ba7d231aceaf9043b394acd343a23e6beac8454d168bfe4a19ad3886fd` |
| `data/knowledge/knowledge_base.json` | `6853d5181a28eb480c1a9c21c180fe261f99701ae1887b94207931cd30c0af94` |
| `data/knowledge/latex_tutorial.json` | `d033db3989cb0d66c9ef4022861e3682b3860147c278f95bccf57a8399cdee67` |
| `data/knowledge/latex_tutorial_pages.json` | `74882f35c4a4aa3795270bff9fbff94a261dfe1434952ef764abe9fb5ae00759` |
| `data/knowledge/obsidian_knowledge.json` | `2f3a8f619377fc184089d2354672610ee74084d251442de365ab00de511be2b4` |

The `data/test.csv` hash matches `PROGRESS.md`. Training, evaluation, calibration, ablation, report artifacts, and `legacy/` are outside this milestone's changes.

## Risks to resolve before implementation

1. **Two sources of truth:** each definition contains v2 and legacy fields. If they disagree, runtime behavior is ambiguous. Define precedence and consistency validation; do not silently regenerate or overwrite either representation.
2. **No executable v2 runtime:** the migration report cannot substitute for checked-in loaders, schema/runtime tests, JSONata, or Jinja sandbox execution.
3. **Missing-value semantics:** distinguish absent, null, zero, false, empty string, optional, defaulted, omitted, asked, and error states. Python truthiness must not collapse them.
4. **Extraction capability claims:** `model` is a contract method, not a configured provider. Keep it unavailable or return a clear unsupported-provider result until an owner configures one.
5. **Classifier/retrieval coordination:** preserve checkpoint label maps, temperature, tau/theta, and fingerprints. A new definition can enter dynamic retrieval without claiming it is a trained PhoBERT class.
6. **Publish atomicity:** file replacement, registry snapshot, SBERT index/cache, and rollback need one defined publish boundary; API readers must never see a partial update.
7. **Template safety:** Jinja must use a sandbox and restricted context. LaTeX still needs existing escaping/static checks; no arbitrary source may reach the compiler.
8. **Migration evidence:** the named source archive is missing, so the migration report's preservation assertions cannot currently be independently reproduced.
9. **Contract gaps:** `domain` is not consistently present; the schema ID is a placeholder; recursive input-schema validation and compatibility rules need to be specified.

## Proposed migration checkpoints

Each milestone is separately reviewable and gated. Do not start KA-16 until the owner approves this plan.

| ID | Deliverable | Exit gate |
|---|---|---|
| KA-16 | Contract loader, schema validation, legacy-to-runtime read adapter, version/precedence policy | Validate all 189 definitions; compare retained legacy fields to a trusted baseline; reject invalid nested schemas with paths; no mutation on load. |
| KA-17 | Recursive schema validation, field normalization, and missing-value policy engine | Scalar/object/array/nested schemas; explicit absent/null/0/false/empty cases; `use_value`, `use_default`, `ask_user`, `omit`, `error`; path-specific errors. |
| KA-18 | Extraction adapter registry: regex, parser, path, manual, and provider interface | Legacy regex parity; dynamic array/nested extraction through configured adapters; model method reports unavailable without provider; no intent/field-name branches. |
| KA-19 | JSONata processor registry | Identity, mapping, filtering, projection, nested access and object/array construction; invalid expression errors; no Python `eval`. |
| KA-20 | Sandboxed Jinja2 and output-renderer registry | Placeholders, conditionals, loops, nested data, multiple templates; sandbox escape tests; preserve LaTeX escape/static checks. |
| KA-21 | Generic pipeline integration with legacy facade | Existing CLI/API/status and 189-definition behavior stay compatible; structured stage results; manual continuation resumes validated prior state. |
| KA-22 | Atomic definition registry publish/reload and SBERT update | New alias/definition retrievable without PhoBERT retraining; cache hash changes correctly; rollback and concurrent snapshot tests. |
| KA-23 | Versioned knowledge-management API | Validate, preview, import/export, publish, reload, backup/rollback; old `/ask` and `/healthz` contracts unchanged. |
| KA-24 | Local GUI dashboard and definition wizard | Dashboard/search/filter and nested schema/policy/transform/render builders call APIs only; no duplicate business logic. |
| KA-25 | GUI test, preview, import/export and publish flow | Create → configure → validate → preview → publish → reload → execute; failure leaves prior definition/index active; user confirmation before overwrite. |
| KA-26 | Integration, documentation, and regression gate | Full legacy suite, contract matrix, GUI/API flow, hash checks, security tests, and documented model/TeX limitations pass. |

Recommended order is KA-16 → KA-17 → KA-18 → KA-19 → KA-20 → KA-21 → KA-22 → KA-23 → KA-24 → KA-25 → KA-26. Keep the old runtime as a compatibility path until KA-21 parity passes; only then consider making the v2 engine the default. Existing schemas and data remain authoritative artifacts until an explicit, tested migration changes them.

## Acceptance executed for KA-15

```text
$ ./AI/bin/python -m core.validator
[Thành công] Tất cả kịch bản đã vượt qua kiểm thực schema nghiêm ngặt!
$ ./AI/bin/pytest tests/test_intent_invariants.py tests/test_audit_regressions.py tests/test_api_server.py -q
58 passed in 8.56s
```

The recorded `data/test.csv` hash matched the actual file, and all four knowledge-corpus hashes matched the baseline table above. An additional attempt to collect `tests/test_ablation_stats.py::test_extract_test_sha256_from_progress_real_file` was blocked before collection because `matplotlib` is missing from the active Python environment (`ModuleNotFoundError`); no dependency was installed. This is outside KA-15's acceptance and does not affect the direct hash check above.

These checks are an audit baseline, not evidence that Contract v2 processors work. No application code, dependencies, definitions, corpora, or tests were changed by KA-15.
