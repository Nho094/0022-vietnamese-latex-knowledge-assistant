# FIX-ALL change log

## Phase 0 — IDs and docs

Acceptance command: `python train/ablation_stats.py --verify-test` (invoked with `/KINGSTON/Github/AI_Agent/AI/bin/python`).

Real output:

```text
======================================================================
VIETNAMESE LATEX KNOWLEDGE ASSISTANT - TEST SET INTEGRITY VERIFIER
Tuân thủ Honesty Rule #3 (AGENTS.md)
  Đường dẫn tệp kiểm thử : /KINGSTON/Github/AI_Agent/projects/0022-vietnamese-latex-knowledge-assistant/data/test.csv
  Mã băm thực tế (Actual) : ef8781ba7d231aceaf9043b394acd343a23e6beac8454d168bfe4a19ad3886fd
  Mã băm quy định (PROGRESS): ef8781ba7d231aceaf9043b394acd343a23e6beac8454d168bfe4a19ad3886fd
  Trạng thái xác thực     : [ĐẠT CHUẨN] HỢP LỆ (VERIFIED)
======================================================================
```

Changes: prompt IDs KA-00..KA-14 replace the active progress table; the prior table is retained in `docs/HISTORY_PROGRESS.md` with an approximate mapping. `AGENTS.md` now has Project and Environment sections. `data/test.csv` was not changed.


## Phase 1 — canonical core

Acceptance: `/KINGSTON/Github/AI_Agent/AI/bin/pytest -q -m 'not slow and not needs_weights'`

Real output: `393 passed, 8 warnings in 101.74s`. Warnings were pyvi deprecation, expected synthetic split warnings, retriever API rename, and scheduler ordering warning. The initial run exposed test defects; fixed hash marker format, subprocess interpreter resolution, disabled unused RoBERTa pooler gradients, and set both roundtrip models to eval mode. `grep -R "from src\|import src" -n --include='*.py' . --exclude-dir=.git --exclude-dir=legacy` returned no matches.

## Phase 2 — validator and intent bank quality

Acceptance output:

```text
$ /KINGSTON/Github/AI_Agent/AI/bin/python -m core.validator
[Validator] Đang kiểm thực: .../data/intents.json (bộ lọc domain: toàn bộ)...
[Thành công] Tất cả kịch bản đã vượt qua kiểm thực schema nghiêm ngặt!
$ /KINGSTON/Github/AI_Agent/AI/bin/pytest tests/test_validator.py -q
14 passed in 0.09s
```

Changes: full-bank validation is default; aliases are checked for digits, repeated unaccented intent-name copies, and cross-intent collisions. Added deterministic future slug function mapping `đ` to `d`; existing IDs were preserved. Added `display_name` to affected IDs; generated aliases for the four underfilled intents are marked `review_needed` and listed in `docs/ALIAS_REVIEW.md`. Removed unused uppercase slots that represented fixed symbols already present in templates.

## Phase 3 — data governance (PARTIAL: owner benchmark pending)

Acceptance commands and real output:

```text
$ /KINGSTON/Github/AI_Agent/AI/bin/python data/build_test_oos.py
Wrote data/test_oos.csv: 27 rows (source candidates=27)
$ /KINGSTON/Github/AI_Agent/AI/bin/python data/check_split_shift.py
Wrote report/split_shift.json
WARNING: val boilerplate share 0.5429 exceeds 0.50.
WARNING: val split-specific boilerplate share 0.9721 exceeds 0.50.
WARNING: test boilerplate share 0.5585 exceeds 0.50.
WARNING: test split-specific boilerplate share 1.0000 exceeds 0.50.
train: n=3825 mean_chars=72.55 boilerplate_share=0.2745 split_boilerplate_share=0.1098 labels=189 OOS=65
val: n=967 mean_chars=158.38 boilerplate_share=0.5429 split_boilerplate_share=0.9721 labels=189 OOS=27
test: n=940 mean_chars=193.05 boilerplate_share=0.5585 split_boilerplate_share=1.0000 labels=188 OOS=0
$ /KINGSTON/Github/AI_Agent/AI/bin/pytest tests/test_dataset.py -q
16 passed in 5.95s
$ sha256sum data/test.csv
ef8781ba7d231aceaf9043b394acd343a23e6beac8454d168bfe4a19ad3886fd  data/test.csv
```

`data/data_report.json` now derives split/label counts from the current CSV files and states the AI-draft status. `data/test_owner_template.csv` has 36 blank rows (3 per each of 12 current categories); `docs/OWNER_TEST_GUIDE.md` documents the owner workflow. `data/verify_owner_test.py` validates schema/labels/category/duplicates/overlap and records its hash only after the owner file exists and passes. Owner test remains BLOCKED until written by the user.

## Phase 4 — threshold wiring

Acceptance: `/KINGSTON/Github/AI_Agent/AI/bin/pytest tests/test_calibrate_tau.py tests/test_threshold_wiring.py tests/test_config.py -q`.

Real output: `34 passed, 2 warnings in 62.92s`. Warnings: SentenceTransformer embedding-dimension API rename and pyvi `codecs.open` deprecation.

The pipeline now loads calibrated thresholds only when checkpoint SHA-256 and preprocessing fingerprint match; otherwise it warns and uses defaults. Smoke checkpoints and zero tau export only to `models/smoke/thresholds_smoke.json`, with `smoke_mode: true`. Calibration binds metadata to classifier checkpoint and preprocessing fingerprint. No real calibration was claimed.

## Phase 5 — artifacts and real CPU baselines (models BLOCKED)

Smoke/stale artifacts were moved, not deleted: ignored local checkpoint directories and `models/thresholds.json` are under `models/smoke/`; tracked report outputs and old confusion figures are under `report/smoke/`. The obsolete checkpoint is verified smoke (hidden size 64, 1 layer, 2 labels); bi-encoder metadata is 64 dimensions without a model identity; PhoBERT cache contains tokenizer files only. No real PhoBERT/bi-encoder weights were found. `scripts/run_real_models.sh` contains owner commands using `vinai/phobert-base-v2` and the configured Vietnamese bi-encoder training path.

Real command outputs (each used train=3825, val=967, test=940 for main test; C selected on val; 1000 bootstrap rounds):

```text
--target intent: best C=0.1; TF-IDF accuracy=0.9617, macro-F1=0.9527; fuzzy accuracy=0.1543, macro-F1=0.1310; hash=ef8781ba7d231aceaf9043b394acd343a23e6beac8454d168bfe4a19ad3886fd
--target category: best C=1.0; TF-IDF accuracy=1.0000, macro-F1=0.9167; fuzzy accuracy=0.4191, macro-F1=0.4140; same hash
```

Separate `data/test_oos.csv` (27 rows, hash `ab10e383dd977c5b0383cb8129738b6cea7cf27f42468d3b5436c31ad8b7344c`): intent TF-IDF accuracy 1.0000, macro-F1 0.0053; category TF-IDF accuracy 0.9259, macro-F1 0.0801. These are classification metrics with OOS labels, not a calibrated system OOS rejection rate. All test.csv metrics are marked `ai_draft` in the aggregate artifact. `python report/build_report.py && pytest tests/test_build_report.py -q`: `1 passed in 0.05s`.

## Phase 6 — KA-10 ablation runner (PARTIAL)

Added a hash-gated runner that writes the required columns and marks every mode BLOCKED because frozen five-seed real predictions/checkpoints are absent. It does not fabricate metrics or overwrite real baseline outputs. `pytest tests/test_ablation_format.py -q`: `1 passed`; broader phase check with CLI/report tests: `9 passed in 22.64s`.

## Phase 7 — CLI/API

Default CLI mode now selects `phobert_retrieval` only for a non-smoke checkpoint, otherwise `retrieval_only` with the visible fallback banner (stderr for one-shot text, terminal for REPL). `kb check` validates the full bank. `kb add` writes `regex`, validates the combined bank, atomically replaces intents and KB state, refreshes the retriever and marks `needs_retrain`.

Acceptance: `/KINGSTON/Github/AI_Agent/AI/bin/pytest tests/test_ka_cli.py tests/test_api_server.py -q` → `8 passed in 22.45s`; toy intent is valid under the full validator and immediately retrievable without classifier training.

## Phase 8 — report

`report/build_report.py` now requires `data/test.csv` SHA-256 to occur in PROGRESS, excludes smoke-folder and mismatched-hash artifacts, and writes the split counts, PhoATIS OOS availability, AI-draft test caveat, and static-only LaTeX status. Acceptance: `python report/build_report.py` → `Đã sinh report/REPORT.md`; `pytest tests/test_build_report.py -q` → `1 passed in 0.06s`.

## Phase 9 — reproducibility and final gate

Pinned the direct dependency set in `requirements.txt`; `requirements.lock` is `pip freeze` from the existing `/KINGSTON/Github/AI_Agent/AI` environment (no packages installed). Added `scripts/verify_all.sh` and Ruff configuration. The final run used `PATH=/KINGSTON/Github/AI_Agent/AI/bin:$PATH scripts/verify_all.sh`.

Real final output summary:

```text
All checks passed!
404 passed, 8 warnings in 99.20s
[Thành công] Tất cả kịch bản đã vượt qua kiểm thực schema nghiêm ngặt!
[ĐẠT CHUẨN] HỢP LỆ (VERIFIED)
No src imports outside legacy/
Wrote report/split_shift.json
WARNING: val boilerplate share 0.5429 exceeds 0.50.
WARNING: val split-specific boilerplate share 0.9721 exceeds 0.50.
WARNING: test boilerplate share 0.5585 exceeds 0.50.
WARNING: test split-specific boilerplate share 1.0000 exceeds 0.50.
train: n=3825 mean_chars=72.55 boilerplate_share=0.2745 split_boilerplate_share=0.1098 labels=189 OOS=65
val: n=967 mean_chars=158.38 boilerplate_share=0.5429 split_boilerplate_share=0.9721 labels=189 OOS=27
test: n=940 mean_chars=193.05 boilerplate_share=0.5585 split_boilerplate_share=1.0000 labels=188 OOS=0
Đã sinh report/REPORT.md
```

Eight warnings are pre-existing/deprecation or intentional fixture warnings: pyvi `codecs.open`, low-sample synthetic test fixtures, SentenceTransformer method rename, and a scheduler-order unit fixture. No failure remains in this gate.

### A1–A10 audit disposition

| Finding | Status | Evidence / remaining owner action |
|---|---|---|
| A1 IDs/docs | fixed | KA-00..KA-14 in PROGRESS; legacy table retained and Project/Environment sections present. |
| A2 duplicate implementations | fixed | `src/` archived under `legacy/src_v0`; imports ported; verify gate finds none outside legacy. |
| A3 validator/bank aliases | mitigated | Full 189-intent validation passes; generated aliases for four intents remain `review_needed` in `docs/ALIAS_REVIEW.md` for owner review. |
| A4 test distribution/owner benchmark | mitigated | `data/test.csv` unchanged with required hash; shift warning recorded; OOS test separated. Owner must author `data/test_owner.csv` using `docs/OWNER_TEST_GUIDE.md`. |
| A5 broken historical slugs | fixed | Existing IDs retained; display names added; future slug maps đ→d. |
| A6 stale thresholds | fixed | Thresholds are hash/fingerprint checked; smoke and zero-tau exports route to smoke path. Real calibration awaits a real classifier. |
| A7 stale/smoke artifacts and model weights | mitigated | Stale artifacts moved under smoke folders; verified PhoBERT cache has tokenizer only and checkpoint is smoke. Owner command: `bash scripts/run_real_models.sh` in a network-enabled environment. |
| A8 missing ablation runner | mitigated | KA-10 format/hash runner exists and accurately marks all real-model modes BLOCKED pending frozen predictions; no unsupported metric is emitted. |
| A9 CLI/Kb schema | fixed | Dynamic default/banner, global KB validation, atomic writes, `regex` serialization, retriever update and `needs_retrain`. |
| A10 unpinned deps/markers | fixed | Direct requirements pinned, working environment frozen, markers registered, verify script added. |

External prerequisites still BLOCKED: run `bash scripts/run_real_models.sh` with network/model access; write `data/test_owner.csv`; install TeX Live and then run `ka ask 'Tạo phương trình bậc hai' --compile` to check actual compilation. The current session had no `pdflatex`, so LaTeX evidence is static-check only.

## KA-08b — regression baseline before changes

The requested `intent_sweep.py` was absent from the checkout (`python intent_sweep.py` initially exited 2 with FileNotFoundError), so I added a small deterministic sweep utility before implementation to establish a comparable baseline.

`/KINGSTON/Github/AI_Agent/AI/bin/python intent_sweep.py` before fixes:

```text
intents=189 latex_unsupported=41
double_escaped_regex=2 ['LATEX_NAP_PACKAGE_CO_TUY_CHON.options', 'LATEX_NAP_PACKAGE_CO_TUY_CHON.package_name']
unanchored_free_text_regex=3 ['LATEX_NAP_PACKAGE.package_name', 'LATEX_CHEN_HINH_ANH.filename', 'LATEX_FIGURE.file']
neutral_required_slot_false_fills=5
quadratic 'x^2 - 3x + 2 = 0': {'a': '1', 'b': '3', 'c': '2'} missing=[] errors=[]
quadratic 'x bình phương trừ 3x cộng 2 bằng 0': {'a': '1', 'b': '3', 'c': '2'} missing=[] errors=[]
quadratic 'trừ x bình phương cộng x cộng 1 bằng 0': {'a': '1', 'b': '1', 'c': '1'} missing=[] errors=[]
```

Test-first output: `/KINGSTON/Github/AI_Agent/AI/bin/pytest tests/test_intent_invariants.py -q` failed during collection because `core.latex_tools.POST_ACTIONS` does not exist yet (ImportError), as expected before implementation. `data/test.csv` was verified before changes: `ef8781ba7d231aceaf9043b394acd343a23e6beac8454d168bfe4a19ad3886fd`.

## KA-08b — implementation and acceptance

Changes: target-aware templates and candidate filtering are data-driven; `ok` requires nonempty rendered code. Validator now checks double-escaped and unanchored free-text slot regexes, validates `absent_means`, and rejects unknown post actions. Fixed the three affected regex fields in `LATEX_NAP_PACKAGE` and `LATEX_NAP_PACKAGE_CO_TUY_CHON`; no other intent regex was reported. Generic slot trimming uses anchor keywords from sibling slots. Math normalization is shared by numeric/math slots and configured vocabularies; neutral prompts no longer fabricate required values. `tidy_math` runs only through the registered `post` action. The prior integral-specific extraction branch remains because the shared parser does not cover all integral phrasing.

The requested schema flag is the only additional intent-bank edit: `absent_means: 0` on quadratic `a`, `b`, and `c`. It is applied only when an explicit equation ending in zero is present. No other intent records were changed. Test set hash remains `ef8781ba7d231aceaf9043b394acd343a23e6beac8454d168bfe4a19ad3886fd`.

Acceptance outputs:

```text
$ /KINGSTON/Github/AI_Agent/AI/bin/python -m core.validator
[Thành công] Tất cả kịch bản đã vượt qua kiểm thực schema nghiêm ngặt!
$ /KINGSTON/Github/AI_Agent/AI/bin/python intent_sweep.py
intents=189 latex_unsupported=41
unsupported_target_resolutions=41
double_escaped_regex=0 []
unanchored_free_text_regex=0 []
neutral_required_slot_false_fills=0
quadratic 'x^2 - 3x + 2 = 0': {'a': '1', 'b': '-3', 'c': '2'} missing=[] errors=[]
quadratic 'x bình phương trừ 3x cộng 2 bằng 0': {'a': '1', 'b': '-3', 'c': '2'} missing=[] errors=[]
quadratic 'trừ x bình phương cộng x cộng 1 bằng 0': {'a': '-1', 'b': '1', 'c': '1'} missing=[] errors=[]
$ pytest tests/test_intent_invariants.py tests/test_validator.py tests/test_latex_tools.py tests/test_preprocess.py tests/test_ka_cli.py tests/test_api_server.py -q
133 passed, 1 warning in 23.17s
$ /KINGSTON/Github/AI_Agent/AI/bin/ruff check .
All checks passed!
$ /KINGSTON/Github/AI_Agent/AI/bin/pytest -q
448 passed, 8 warnings in 99.90s
```

`pdflatex` is unavailable, so the LaTeX result is static-check only. Unsupported math forms include equations without a recognizable variable/polynomial structure, ambiguous spoken coefficient order, and number words outside the supported Vietnamese 0–99 vocabulary; these remain clarification cases. `intent_sweep.py` shows the requested regressions resolved: 0 bad regexes and 0 neutral false fills (baseline was 2, 3, and 5 respectively); 41 intents remain unsupported at the LaTeX target by design because they lack LaTeX/MathJax templates.

## Repository audit and generic refactor

The broader audit found additional source-data mismatches after KA-08b: sum/root/fraction/limit slots were catch-all text despite mathematical values; the 2x2 matrix scenario lacked query wording; several MathJax/Obsidian templates stored literal `\\n`; `normalize_math` rejected safe commands such as `\\sin`; TF-IDF compared digit queries against aliases written as Vietnamese number words; Ctrl+C escaped the CLI loop; and retrieval called the deprecated SentenceTransformer dimension API first.

Changes made:

- `data/intents.json`: replaced catch-all math slots for sum, square root, fraction, limit, and integral with typed, keyword-anchored patterns; changed 2x2 matrix element slots to math; added the generic declarative `not_equal` constraint for the quadratic leading coefficient; added matrix aliases; converted literal `\\n` separators to actual line breaks across templates. No knowledge records or `data/test.csv` were removed/overwritten.
- `core/slots.py`: removed the intent-name-keyed quadratic validation hook and the integral-specific extraction branch. Numeric constraints are now handled by schema fields; integral/sum/math capture uses the same slot regex engine. Math normalization preserves safe LaTeX commands and text macro spacing and rejects forbidden commands.
- `core/preprocess.py` and `core/pipeline.py`: protected LaTeX commands during spoken-math normalization and canonicalized standalone numeric forms for TF-IDF retrieval without changing exponents, coefficients, or decimal numbers.
- `core/validator.py`: validates numeric `not_equal` constraints and detects literal newline escapes in every target template.
- `core/retrieval.py`: prefers `get_embedding_dimension()` and keeps the old method as compatibility fallback.
- `cli/ka.py`: Ctrl+C reports a short cancellation message and returns exit 130.
- `scripts/verify_all.sh`: excludes the repository's `AI/` runtime environment from source-import scanning. Its first run correctly exposed this scan bug after tests, validator and hash checks had passed; the corrected run passed.
- Added `tests/test_audit_regressions.py` and documented the additive slot fields in `docs/INTENT_SCHEMA.md`.

Verification:

```text
$ pytest -q
461 passed, 7 warnings in 107.46s (final run)
$ PATH=/KINGSTON/Github/AI_Agent/AI/bin:$PATH MPLCONFIGDIR=/tmp/ka-mpl-cache scripts/verify_all.sh
Ruff: All checks passed!
pytest subset: 461 passed, 7 warnings in 109.98s
Validator: all 189 intents passed
test.csv: ef8781ba7d231aceaf9043b394acd343a23e6beac8454d168bfe4a19ad3886fd [VERIFIED]
No src imports outside legacy/
Split shift report written; existing val/test boilerplate warnings remain
report/REPORT.md generated
```

Seven test warnings remain: PyVi's upstream `codecs.open()` deprecation, intentional low-sample synthetic split fixtures, and a scheduler-order unit fixture. The SentenceTransformer dimension deprecation warning no longer occurs. No `pdflatex` compiler result is claimed; static LaTeX checking only. Existing model-weight and owner-benchmark blockers are unchanged.
