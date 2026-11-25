# Prompt Pack: Knowledge Assistant (Half Vibe Code) - bản LaTeX

*Bộ prompt chạy từng bước cho AI Agent, dựa trên 5 bài báo (RAF, PhoBERT, JointBERT, BiJoint-BERT-NLU, Sentence-BERT) và bản kế hoạch dự án. Bản này sinh mã LaTeX thay cho Python; phần mô hình và phương pháp giữ nguyên. Cập nhật 19/11/2026.*

## 0. Cách dùng: ID + resume

1. Dán mục 1 (MASTER) vào file AGENTS.md (hoặc CLAUDE.md, tùy agent) ở gốc repo. Agent đọc nó mỗi phiên.
2. Mỗi lần chỉ dán MỘT prompt KA-xx. Agent làm, tự kiểm tra, ghi PROGRESS.md rồi dừng.
3. Lệnh điều khiển (gõ cho agent): STATUS, RESUME, RESUME KA-05, REDO KA-05, EXPLAIN KA-05 (giải thích bằng tiếng Việt file vừa viết, không sửa code).
4. Phiên mới hoặc agent bị ngắt: dán đúng câu này: Read AGENTS.md and PROGRESS.md, then RESUME.
5. Bộ test do bạn tự viết tay và chốt TRƯỚC khi chạy KA-05. Agent không được sửa hay sinh test.csv.

| ID | Việc | Phụ thuộc | Bài báo | Loại |
| --- | --- | --- | --- | --- |
| KA-00 | Scaffold, PROGRESS.md, config | - | - | Bắt buộc |
| KA-01 | Schema intents.json + validator + chuỗi scenario + latex_tools | 00 | RAF | Bắt buộc |
| KA-02 | Dữ liệu train/val/test + câu ngoài phạm vi | 01 | RAF, PhoATIS (BiJoint) | Bắt buộc |
| KA-03 | Tiền xử lý + tách từ | 00 | PhoBERT | Bắt buộc |
| KA-04 | Baseline TF-IDF + SVM + bộ metric | 02, 03 | JointBERT (cách đo) | Bắt buộc |
| KA-05 | PhoBERT phân loại intent (mô hình chính) | 02, 03, 04 | PhoBERT, JointBERT Eq.1 | Bắt buộc |
| KA-06 | Truy xuất ngữ nghĩa (inference + đo) | 01, 03 | SBERT, RAF | Bắt buộc |
| KA-07 | Fine-tune bi-encoder (contrastive, triplet) | 02, 06 | RAF, SBERT | Mở rộng |
| KA-08 | Slot (regex) + renderer + test template (biên dịch LaTeX) | 01 | RAF (Fill), JointBERT | Bắt buộc |
| KA-09 | Pipeline + ngưỡng tau/theta | 05, 06, 08 | RAF | Bắt buộc |
| KA-10 | Đánh giá + ablation | 09 | tất cả | Bắt buộc |
| KA-11 | Joint intent + slot BIO | 05, 08 | JointBERT Eq.2-3 | Mở rộng |
| KA-12 | BiJoint-BERT-NLU | 11 | BiJoint | Mở rộng |
| KA-13 | CLI + Knowledge Manager + API tối thiểu | 09 | - | Bắt buộc (CLI) |
| KA-14 | Báo cáo | 10 | - | Bắt buộc |

Gợi ý chia phiên: phiên 1 = KA-00 đến KA-03; phiên 2 = KA-04, KA-05; phiên 3 = KA-06, KA-08; phiên 4 = KA-09, KA-10, KA-13; sau đó mới mở rộng KA-07, KA-11, KA-12.

### Bốn chỗ công thức cần xử lý khi cài (đã đưa vào prompt)

1. RAF Eq.(1) in ra log-tỉ-số không có dấu trừ. Khi huấn luyện phải cực tiểu hóa log-likelihood ÂM.
2. BiJoint: mẫu số của attention in là tổng exp(w^T h\_i) với i cố định, đó là lỗi in; đúng là tổng theo j của exp(w^T h\_j).
3. BiJoint: ký hiệu vòng tròn trong v\_i và \[q, h\_0\] được văn bản mô tả là concatenation (nối vector), không phải nhân từng phần tử.
4. Đầu phân loại RoBERTa mặc định của Hugging Face (theo hiểu biết của mình, agent phải tự kiểm tra) có thêm lớp dense + tanh, khác Eq.(1) của JointBERT là một lớp tuyến tính trên h\_0. Prompt KA-05 yêu cầu tự dựng đầu đúng công thức.

Các file PDF bạn gửi là bản trích (RAF đến mục 4.1, PhoBERT trang 1-7, JointBERT trang 3-5, SBERT trang 1-7, BiJoint từ mục 2 đến kết luận). Phần không có trong bản trích (phụ lục siêu tham số của RAF, bảng ablation của SBERT) không được đưa vào prompt; agent không được tự bịa các con số đó.

### Khác biệt so với bản Python (đã đưa vào prompt)

1. Đích sinh mã đổi từ Python sang LaTeX: khóa `latex` là template chính, `python` và `cpp` để trống; `TARGET = 'latex'`.
2. Chỗ trống trong template đổi từ `{slot}` sang `<<slot>>`, vì LaTeX dùng dấu ngoặc nhọn ở khắp nơi (`\section{...}`). Điền bằng đúng một lượt `re.sub`, không dùng `str.format`.
3. Giá trị slot kiểu `text` phải được escape các ký tự đặc biệt của LaTeX (`\ { } $ & # ^ _ ~ %`) trong một lượt thay thế duy nhất. Nhờ vậy câu của người dùng không chèn được lệnh vào mã sinh ra.
4. Cách kiểm tra template thay `ast.parse` và chạy thử bằng: (a) kiểm tra tĩnh (ngoặc nhọn cân bằng, `\begin`/`\end` khớp, cấm các lệnh như `\write`, `\input`) và (b) biên dịch thử bằng `pdflatex` (không shell-escape, có timeout). Nếu máy chưa có TeX Live, agent không được tự cài: bước (b) bị bỏ qua và phải ghi rõ "chưa biên dịch thử".
5. Danh mục intent đổi từ `py_*` sang `tex_*` (xem KA-01), theo các chủ đề của bài Overleaf "Learn LaTeX in 30 minutes": cấu trúc tài liệu, định dạng chữ, danh sách, bảng, hình, công thức, tham chiếu, bố cục.
6. Tiếng Việt trong LaTeX: khung bọc để biên dịch thử dùng `[T5]{fontenc}` và `[vietnamese]{babel}` (pdflatex).
7. Các phần không phụ thuộc đích sinh mã (KA-03 đến KA-07, KA-09 đến KA-12) giữ nguyên, chỉ sửa chỗ nào nhắc tới Python.

## 1. MASTER prompt (lưu thành AGENTS.md)

```text
# Knowledge Assistant (Half Vibe Code) - AGENT RULES

## Project
Local Vietnamese knowledge assistant. The user types a Vietnamese sentence. The system (1) picks a scenario (intent + template) from a user-managed scenario bank (data/intents.json), (2) extracts slot values, (3) fills a LaTeX template and returns LaTeX code. Only the latex template is in scope now; keep the keys python and cpp in the schema but leave them empty.
Academic goal: study and evaluate ONE model: the RoBERTa family for Text Classification (project #5 of 11). PhoBERT (pre-training approach based on RoBERTa) is the Vietnamese encoder used for intent classification. Sentence-BERT-style retrieval is the fallback and handles NEW intents without retraining. Slot filling in the core pipeline is regex-based.
Environment: Linux (Gentoo), Python 3.11+, CPU must work, GPU optional. Cache models under models/hf_cache. Do not assume internet after dependencies and models are downloaded. LaTeX compile tests need pdflatex (TeX Live); it is optional: when it is missing, skip the compile tests with an explicit reason and never install it.

## Architecture (do not change without asking)
text -> preprocess (NFC, math symbol normalization, pyvi word segmentation for PhoBERT)
  -> PhoBERT classifier: p = softmax over intents (or over categories in two-tier mode)
     - if max p >= tau: intent accepted
     - else retrieval top-3 over aliases + scenario strings -> ask the user to choose (if best score < theta: say not understood)
  -> slot extraction (regex per intent) -> if a required slot is missing: ask for exactly the missing slots
  -> render latex template (text slots escaped) -> output

## Honesty rules (very important)
1. The filling step is regex, far simpler than the span-pointer decoder of Retrieve-and-Fill (RAF). Always describe the project as a simplified scenario-based system inspired by RAF, never as a re-implementation.
2. Never invent a number. Every metric in any report must come from a script output saved under report/. If you cannot run something, say so.
3. Never tune on the test set. Evaluate test once per final configuration. Store sha256 of data/test.csv in PROGRESS.md at KA-05 and verify it before every final evaluation. Never edit, generate or augment test.csv.
4. Train/val/test must not share duplicate sentences (case and whitespace insensitive).
5. Numbers from the papers (e.g. PhoATIS results) may only be quoted as reported in the paper, never as our results.
6. If a paper formula is ambiguous or looks like a typo, implement the interpretation given in the task prompt and note it in the docstring.
7. Never say a template compiles unless the compile test ran and passed in this session; if only the static check ran, say 'static check only'.

## Code rules
- Python 3.11, type hints, ruff-clean, pytest for every module under tests/.
- Identifiers in English; docstrings and comments in Vietnamese explaining WHY and citing the paper formula (Half Vibe Code: the owner must understand every file).
- Config only in core/config.py; no magic numbers elsewhere. Seeds via one helper (random, numpy, torch).
- Do not edit or delete legacy/. The old files core/matcher.py, core/parser.py, core/math_extractor.py are kept as the fuzzy baseline: import them, do not rewrite them unless a task says so.
- Do not install system packages (TeX Live included: ask the owner); use a venv. Pin versions in requirements.txt after the first successful install.
- Never run LaTeX with shell escape (always -no-shell-escape) and never compile text that did not come from the template bank.
- One task per session: do only the requested ID.

## Progress protocol
- PROGRESS.md holds a table: ID | title | status (TODO/IN_PROGRESS/DONE/BLOCKED) | files | acceptance command | last result | notes, plus sections Decisions and Open questions.
- At session start read AGENTS.md and PROGRESS.md.
- STATUS: print the table. RESUME: continue the first ID that is not DONE. RESUME <ID>: continue that ID. REDO <ID>: set it TODO and redo. EXPLAIN <ID>: explain in Vietnamese what the files do and why; no code changes.
- At task end: run the acceptance commands, report the real output, update PROGRESS.md, reply in at most 10 lines (done, failed, what the owner must decide) and STOP. Do not start the next ID.
- If a task is too big for one session, split it, mark the finished part, list the remaining steps in the notes.
- Ask the owner (do not guess) when a dataset licence is unclear, a model name does not exist, or a change affects the architecture.
```

## 2. KA-00: Scaffold

```text
TASK KA-00: Repository scaffold. Depends on: nothing.
Goal: make the repo match the target layout without losing existing work.
Current tree (keep everything): api/server.py, cli/ka.py, core/{classifier,matcher,math_extractor,parser,pipeline,preprocess,renderer,retrieval,slots}.py, data/knowledge_base.txt, legacy/{deploy,main.py,web}, modules/, packing/gentoo, report/, requirements.txt, test/test_template.py, train/{evaluate,join_intent_slot,triplet_sbert}.py, README.md.
Do:
1. Print the current tree and ask for confirmation before applying these renames: test/ -> tests/, test_template.py -> test_templates.py, packing/ -> packaging/, train/join_intent_slot.py -> train/joint_intent_slot.py. Use git mv if the repo is under git.
2. Create dirs: data/external, models/hf_cache, report/figures, prompts/, docs/. Add .gitignore for models/, data/external/, __pycache__, .venv, *.pt, *.safetensors.
3. Create core/config.py (pathlib) as the single source of truth: ROOT, DATA_DIR, MODELS_DIR, REPORT_DIR, INTENTS_PATH, PHOBERT_NAME='vinai/phobert-base-v2', EMBED_MODEL_NAME='bkai-foundation-models/vietnamese-bi-encoder' (verify both names exist on the Hub before relying on them; if one fails, STOP and ask), MAX_LEN=64, SEEDS=[13,42,2024,7,99], TAU=0.7, THETA=0.35, TOP_K=3, TWO_TIER=False, TARGET='latex', LATEX_ENGINE='pdflatex', LATEX_TIMEOUT=30, FIGURE_EXTS=('png', 'jpg', 'jpeg', 'pdf'), SEGMENTER='pyvi', LOWERCASE_PHOBERT=False. Add set_seed(seed) seeding random, numpy, torch and enabling deterministic flags when available.
4. requirements.txt without pins: torch, transformers, sentence-transformers, scikit-learn, pandas, numpy, pyvi, joblib, matplotlib, pytest, ruff, fastapi, uvicorn, datasets, seqeval, rapidfuzz.
5. Create PROGRESS.md with the table KA-00..KA-14 (titles as in the prompt pack, all TODO except this one) and empty sections Decisions and Open questions.
6. README.md: 3-command venv setup. tests/test_config.py: paths exist; set_seed gives identical torch.rand for two calls.
7. Check once whether the LaTeX engine is on PATH (shutil.which(config.LATEX_ENGINE)) and write the result in PROGRESS.md under Decisions. Do not install TeX Live.
Acceptance: python -c 'import core.config' works; pytest tests/test_config.py passes; ruff check passes; git status shows no deleted legacy files.
Do NOT: train anything, download datasets, or touch legacy/.
```

## 3. KA-01: Schema, validator, scenario string

```text
TASK KA-01: intents.json schema, validator, scenario string, LaTeX tools. Depends on: KA-00.
Background (Retrieve-and-Fill): a SCENARIO is an intent-slot template with variable leaf spans, e.g. [IN:GET_WEATHER [SL:LOCATION x1]]. All supported scenarios are known in advance (the scenario bank). Our data/intents.json is that bank. RAF scenarios are not one-size-fits-all: a scenario explicitly lists the slots it needs, and a sentence lacking a required span is not compatible with it. Our deliberate deviation (write it in docs/DECISIONS.md): slots that are only names or captions may be required=false with a default value.
Schema of one intent object (data/intents.json is a JSON list):
- intent: UPPER_SNAKE_CASE, unique
- category: one of tex_doc (document class, preamble, packages, Vietnamese setup), tex_struct (title, abstract, sections, table of contents, appendix), tex_text (bold, italic, underline, font size, alignment, quotes, footnotes, line breaks), tex_lists (itemize, enumerate, description), tex_tables (tabular, table float, caption), tex_figures (includegraphics, figure float, caption), tex_math (inline math, equation, align, fractions, sums, matrices), tex_refs (label, ref, cite, bibliography, hyperlinks), tex_layout (margins, spacing, page numbers, columns, page breaks), tex_misc (comments, special characters, verbatim) (OUT_OF_SCOPE is a special intent with empty aliases/slots/templates)
- subcategory: optional snake_case string
- describe: Vietnamese, form '<name> | <slots or keywords>'
- aliases: 5-8 short Vietnamese phrases that NAME the task (no digits, no near-duplicates)
- slots: {name: {type: number|text|label|filename, regex: exactly one capture group, required: bool, default: str|null, desc: optional Vietnamese description, examples: optional list of example values}}. The type decides normalisation and escaping in KA-08: number = a numeric literal; text = free text, escaped for LaTeX; label = ASCII [a-z0-9:_-]+ used as a \label or \ref key; filename = ASCII [A-Za-z0-9_./-]+ (no '..', no leading '/').
- templates: {latex: LaTeX code with <<slot>> placeholders; python and cpp keys allowed but empty}. Placeholders are written <<slot>>, never {slot}: LaTeX code is full of braces.
- kind: snippet (goes inside the document body), preamble (goes before \begin{document}) or document (a complete .tex file)
- requires: list of LaTeX packages the template needs (for example amsmath, graphicx); compilable: bool
- post: optional, null or 'tidy_math' (a clean-up rule applied after filling, defined in KA-08)
Scenario string R(s) (RAF rewrites scenarios as natural strings so the encoder understands the meaning of intent and slot names, e.g. [ intent | get weather [ slot | location x1 ] ]). Implement core/scenario.py: scenario_string(intent, variant) with variants:
- name: '[ intent | <name part of describe> ]'
- slots: name + '[ slot | <desc or slot name> ]' for each slot
- primed: slots + up to 2 example values per slot after a bar, e.g. '[ slot | don vi nhiet do | F / C ]' (RAF example priming)
- handmade: the full primed string; the ONLY variant used at inference
Training later samples uniformly among the variants (RAF representation sampling).
core/latex_tools.py (pure functions, used by the validator, the renderer and the CLI):
- escape_text(s): maps \ -> \textbackslash{}, { -> \{, } -> \}, $ -> \$, & -> \&, # -> \#, ^ -> \textasciicircum{}, _ -> \_, ~ -> \textasciitilde{}, % -> \%, in ONE pass with re.sub and a dict (chained str.replace would escape the backslashes it has just inserted).
- static_check(code) -> list[str] of problems: braces balanced and the depth never negative (ignore \{ and \}, ignore text after an unescaped %); every \begin{env} closed by the matching \end{env} in the right order; no FORBIDDEN control word. FORBIDDEN = input, include, write, openout, openin, read, immediate, catcode, directlua. Match WHOLE control words with the regex \\(?:input|include|write|openout|openin|read|immediate|catcode|directlua)(?![A-Za-z]) so that \includegraphics stays allowed.
- wrap_snippet(code, requires, kind) -> str: kind document returns the code unchanged; otherwise build, in this order: \PassOptionsToPackage{draft}{graphicx} (so a missing image file does not stop compilation); \documentclass[11pt]{article}; \usepackage[utf8]{inputenc}; \usepackage[T5]{fontenc}; \usepackage{lmodern}; \usepackage[vietnamese]{babel}; one \usepackage{pkg} per required package; for kind preamble the code goes here; \begin{document}; for kind snippet the code goes here; \end{document}. If the T5/babel combination fails on the installed TeX, record the exact log line in PROGRESS.md and ask the owner; do not guess a fix.
- compile_check(code, requires, kind) -> (status, log_tail) with status in {ok, failed, skipped}. skipped when shutil.which(config.LATEX_ENGINE) is None. Otherwise write the wrapped text to main.tex inside a tempfile.TemporaryDirectory and run [engine, '-interaction=nonstopmode', '-halt-on-error', '-no-shell-escape', 'main.tex'] with cwd = that directory and timeout = config.LATEX_TIMEOUT; failed on a non-zero exit or a timeout; log_tail = the last 20 log lines.
Write data/validate_intents.py (argparse: files..., --out). Collect ALL errors, do not stop at the first:
1. keys and types are correct (slot type and kind are in the allowed lists). 2. intent names unique. 3. category allowed. 4. aliases 5-8, no digits, no duplicates inside one intent, and NO alias shared by two different intents (case and whitespace insensitive). 5. every regex compiles and has exactly 1 capture group; required=false needs a non-null default. 6. the template passes latex_tools.static_check after replacing each <<slot>> with its default or a dummy by type (number 1, text abc, label lbl, filename img.png), using ONE re.sub pass. 7. every leftover token <<name>> (lowercase identifier) must be a declared slot; a declared slot never used by the template is a warning. 8. if compilable is true: compile_check; failed is an error, skipped is a WARNING and the report lists the intent as 'not compiled'. 9. OUT_OF_SCOPE is allowed with empty aliases/slots/templates.
Output: error table grouped by intent; write the merged file only when there are zero errors; exit code 1 otherwise; save counts per category and the number of templates not compiled to report/intent_validation.txt.
Seed the file by hand with 3 intents so the validator has data:
- QUADRATIC_EQUATION: category tex_math, kind snippet, requires [amsmath], post tidy_math; slots a, b, c (type number, required). Template:
  \begin{equation}
    <<a>>x^{2} + <<b>>x + <<c>> = 0
  \end{equation}
- LATEX_FIGURE: category tex_figures, kind snippet, requires [graphicx]; slots file (type filename, required), caption (type text, default 'Hinh minh hoa'), label (type label, default 'fig:hinh'), width (type number, default 0.8). Template:
  \begin{figure}[htbp]
    \centering
    \includegraphics[width=<<width>>\textwidth]{<<file>>}
    \caption{<<caption>>}
    \label{<<label>>}
  \end{figure}
- OUT_OF_SCOPE.
Tests tests/test_validate.py: one valid file passes; each error class above is triggered by a crafted bad intent (including an unbalanced brace, a mismatched \begin/\end, a template containing \immediate\write18{ls}, and a slot used in the template but not declared). tests/test_scenario.py: the four variants for QUADRATIC_EQUATION. tests/test_latex_tools.py: escape_text on '50% & #1_a {x} ~ ^ \ $' gives the expected string; static_check flags '\begin{a}\end{b}', a closing brace with no opening one, and \write, but accepts \includegraphics; compile_check on both seeded templates (skipped with an explicit reason when the engine is missing).
Acceptance: python data/validate_intents.py data/intents.json passes; pytest tests/test_validate.py tests/test_scenario.py tests/test_latex_tools.py pass.
Do NOT: generate the large knowledge base here (it arrives later as external batches); put \usepackage inside a snippet template (packages go in requires).
```

## 4. KA-02: Dữ liệu

```text
TASK KA-02: dataset. Depends on: KA-01.
Goal: data/train.csv, val.csv, test.csv with columns text,label,category (category derived from intents.json) and a leakage report.
Sources (record provenance in data/SOURCES.md; ask the owner if any licence is unclear):
a) data/raw_sentences.csv written by the owner (columns text,label,split). Owner-written TEST sentences are authoritative and are never modified.
b) Synthetic TRAIN sentences: data/synth_sentences.py with SENTENCE_FRAMES per intent (8-10 Vietnamese frames with {slot} placeholders, plus casual frames without numbers). Fill number slots with random small integers (-9..9, no zero leading coefficient), text slots from a small seeded list of Vietnamese phrases (captions, titles), filename slots from a seeded list (so_do.png, ket_qua.pdf); vary starters (viet, tao, chen, cho minh, giup minh, hay, lam sao de) and put the word latex in some frames. Seeded and deduplicated. TRAIN ONLY.
c) OUT_OF_SCOPE: PhoATIS (Vietnamese ATIS, flight domain, released by VinAI for research or educational use). Download the syllable-level seq.in files PhoATIS/syllable-level/{train,dev,test}/seq.in from https://github.com/VinAIResearch/JointIDSF into data/external/phoatis (skip if present). Sample (seeded) 60/15/20 lines for train/val/test and add about 10 owner-written casual sentences. If there is no network: STOP and tell the owner to download the files manually.
d) Optional val candidates for the tex_math intents from a Hugging Face math dataset (check its licence first): write data/mined_candidates.csv for MANUAL review only; never auto-label.
Split rules: test = owner-written only. val = owner-reviewed sentences with different phrasing from train. Compare duplicates after core.preprocess.normalize (if KA-03 is not done: lowercase and collapse spaces).
Implement data/build_dataset.py: validate labels against intents.json; write the three CSVs; print counts per split and label; fail if a test sentence duplicates a train or val sentence; warn if an intent has fewer than 10 train or fewer than 5 test sentences.
Leakage report report/data_report.txt: counts, exact-duplicate check, average length, and for each test sentence the maximum Jaccard similarity of character trigram sets with any train sentence: J(A,B) = |A intersect B| / |A union B|; flag J > 0.8 as near-duplicate.
Acceptance: python data/build_dataset.py runs; the report shows 0 exact duplicates across splits; pytest tests/test_dataset.py (schema, no overlap) passes.
Do NOT: use an LLM to write the test set; do not scrape websites.
```

## 5. KA-03: Tiền xử lý

```text
TASK KA-03: preprocessing. Depends on: KA-00.
Background (PhoBERT paper): in Vietnamese the space separates SYLLABLES, not words. The paper's example: the 6-syllable text 'Toi la mot nghien cuu vien' forms 4 words 'Toi la mot nghien_cuu_vien'. PhoBERT was pre-trained on word-segmented text (RDRSegmenter from VnCoreNLP) followed by BPE with a 64K vocabulary (about 24.4 subwords per sentence on average; maximum length 256 at pre-training). Un-segmented input silently lowers accuracy, so train-time and run-time preprocessing MUST be identical.
Implement core/preprocess.py:
- normalize(text): NFC; strip; collapse whitespace; map math look-alikes (superscript 2 and 3 to ^2 and ^3, unicode minus and en dash to '-', multiplication sign to '*', division sign to '/', full-width digits to ASCII); keep digits and the symbols + - = ^ * / ( ) x \ { } _ $ % & untouched (a user may type a LaTeX command inside the sentence). Do NOT remove diacritics. Do NOT lowercase when LOWERCASE_PHOBERT is False (the paper does not state that PhoBERT input is lowercased; the TF-IDF baseline lowercases on its own).
- segment(text): pyvi.ViTokenizer.tokenize on non-math parts only; words joined by underscore. Math spans (regex: expressions containing digits and x, operators, equals sign) are protected from the segmenter. LaTeX command spans (regex \\[A-Za-z]+) are protected too. Optional backend 'rdr' (py_vncorenlp, needs Java) behind a flag; if missing raise a clear error; do not install Java.
- prepare_for_phobert(text) = segment(normalize(text)); prepare_for_embedding(text) = the same function by default (the Vietnamese bi-encoder is PhoBERT-based; verify its expected input at KA-06).
- preprocess_fingerprint(): hash of (SEGMENTER, pyvi version, normalization rules), stored inside every trained checkpoint and checked at load time to catch train/serve mismatch.
Tests tests/test_preprocess.py: normalize is idempotent; a multi-word sentence contains underscores after segment; '2x² + 3x - 5 = 0' becomes '2x^2 + 3x - 5 = 0' with the math span untouched by the segmenter; the fingerprint changes when SEGMENTER changes; a sentence containing \includegraphics keeps that command untouched after segment.
Also data/show_tokens.py: for 5 sample sentences print raw, segmented, PhoBERT subword tokens and the number of subwords (to check against MAX_LEN=64).
Acceptance: pytest tests/test_preprocess.py passes; python data/show_tokens.py prints the table.
```

## 6. KA-04: Baseline và metric

```text
TASK KA-04: baseline and metrics. Depends on: KA-02, KA-03.
Implement train/evaluate.py (reused by every later task):
- accuracy = (1/N) * sum_i [pred_i == y_i]
- per class c: P_c = TP_c/(TP_c+FP_c); R_c = TP_c/(TP_c+FN_c); F1_c = 2*P_c*R_c/(P_c+R_c), defined as 0 when the denominator is 0.
- macro-F1 = (1/C) * sum_c F1_c over ALL labels of intents.json (not only labels present); also report weighted-F1.
- confusion matrix PNG (matplotlib only) to report/figures/.
- topk_acc = (1/N) * sum_i [y_i in top-k of scores_i]; MRR = (1/N) * sum_i 1/rank_i.
- mean_std(values) with the SAMPLE standard deviation (n-1).
- bootstrap_ci(y_true, y_pred, metric, B=1000, alpha=0.05, seed): percentile interval.
- mcnemar(y_true, pred_a, pred_b): b = #(a right, b wrong), c = #(a wrong, b right); chi2 = (|b-c|-1)^2/(b+c) with 1 degree of freedom; if b+c == 0 return p = 1.
- ece(confidences, correct, M=10): ECE = sum_{m=1..M} (|B_m|/N) * |acc(B_m) - conf(B_m)| using equal-width confidence bins.
Implement train/baseline_tfidf_svm.py:
- text -> core.preprocess.segment(normalize(text)) lowercased (baseline only) -> TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True) -> LinearSVC(C).
- tf-idf(t,d) = tf(t,d) * idf(t), idf(t) = ln((1+n)/(1+df(t))) + 1, rows L2-normalised (scikit-learn defaults). LinearSVC objective (one-vs-rest, squared hinge): min_w 0.5*||w||^2 + C * sum_i max(0, 1 - y_i * w.x_i)^2.
- Choose C in [0.1, 1, 10] on VAL only; fit on train; report TEST once.
- Second baseline: fuzzy matching with the legacy core/matcher.py if it exposes a callable; otherwise rapidfuzz token_set_ratio against aliases (document which one was used).
- Option --target category trains on the category column.
- Save report/baseline_results.json (accuracy, macro-F1, per-class, bootstrap CI) and the confusion matrix.
Acceptance: python train/baseline_tfidf_svm.py prints the metrics; pytest tests/test_evaluate.py checks every formula on a tiny hand-computed example (the example is written in the test).
Do NOT: look at TEST results while choosing C.
```

## 7. KA-05: PhoBERT phân loại intent (mô hình chính)

```text
TASK KA-05: PhoBERT intent classifier (MAIN MODEL). Depends on: KA-02, KA-03, KA-04.
Why PhoBERT: it is pre-trained with the RoBERTa recipe (an optimised BERT pre-training) on about 20GB of word-segmented Vietnamese text (about 1GB Wikipedia + 19GB news), BPE vocabulary 64K; the base model has about 135M parameters.
Model (JointBERT, intent branch only). For the segmented input with special tokens the encoder output is H = (h_0, h_1, ..., h_T); h_0 is the hidden state of the first special token (<s> in PhoBERT, the [CLS] of BERT). Intent probabilities:
  y_hat = softmax(W h_0 + b),  W in R^{K x 768},  b in R^K,  K = number of labels (intents, or categories when TWO_TIER)
  Loss (cross-entropy): L = -(1/B) * sum_{n=1..B} log y_hat[n, gold_n]
IMPORTANT: to my knowledge the stock Hugging Face RoBERTa classification head has an extra dense + tanh layer, which is NOT the formula above. Print the stock head class once to verify, then build a custom module: AutoModel (no head) + Dropout(0.1) + Linear(768, K) applied to h_0, so the code matches Eq.(1) of JointBERT. Explain the difference in the docstring.
Training protocol (PhoBERT fine-tuning section):
- AdamW, fixed learning rate 1e-5, batch size 32, up to 30 epochs; evaluate on VAL after every epoch; early stopping when VAL macro-F1 has not improved for 5 consecutive epochs; keep the best checkpoint and report the TEST result of that checkpoint; every reported score is the MEAN over 5 runs with different seeds (config SEEDS) together with the sample std.
- JointBERT used lr 5e-5, batch 128, dropout 0.1, max length 50: reference only.
- Our data is small, so allow a learning-rate search over {1e-5, 2e-5, 5e-5} chosen by mean VAL macro-F1 over 2 seeds; then FREEZE the choice before touching test. A linear warm-up over 10% of the steps then linear decay is allowed (our addition; SBERT used a 10% warm-up). max_length = MAX_LEN (64), dynamic padding.
- Mixed precision only when CUDA is available. Provide --smoke (64 train sentences, 1 epoch) so the code path can be verified quickly on CPU.
Files:
- train/finetune_phobert.py: args --target {intent,category} --lr --seeds --smoke. Saves models/phobert_<target>_seed<seed>/ (weights, label2id.json, preprocess_fingerprint, config) and report/phobert_results.json (per-seed and mean +/- std of accuracy and macro-F1, per-class report, confusion matrix, ECE from max-softmax confidence, time per epoch). Compute sha256 of data/test.csv and store it in PROGRESS.md.
- core/classifier.py: class IntentClassifier(model_dir) with predict(text) -> list[(label, prob)] sorted by probability, applying prepare_for_phobert; it must verify that the stored fingerprint equals the current one and raise otherwise; predict_batch(list); a temperature hook (T=1.0 default). Confidence = max probability.
- Compare with the KA-04 baseline using McNemar and the bootstrap CI from train/evaluate.py.
Acceptance: python train/finetune_phobert.py --smoke runs end to end on CPU; pytest tests/test_classifier.py checks (a) probabilities sum to 1, (b) a fingerprint mismatch raises, (c) predict equals predict_batch on the same input.
Do NOT: use the stock sequence-classification head; skip word segmentation; report the best seed instead of the mean.
```

## 8. KA-06: Truy xuất ngữ nghĩa (inference)

```text
TASK KA-06: semantic retrieval (inference + evaluation). Depends on: KA-01, KA-03.
Background (Sentence-BERT). A cross-encoder BERT needs one forward pass per PAIR: finding the closest pair among n = 10,000 sentences takes n(n-1)/2 = 49,995,000 passes (about 65 hours on a V100). A bi-encoder embeds every sentence once (10,000 embeddings, about 5 seconds) and compares vectors by cosine (about 0.01 s). Using the plain average of BERT outputs or the [CLS] vector as sentence embedding is poor (average STS 54.81 and 29.19, even below averaged GloVe at 61.32); SBERT fine-tunes siamese networks. SBERT's default pooling is MEAN.
Formulas:
- mean pooling with attention mask m_t in {0,1}: u = (sum_t m_t * h_t) / (sum_t m_t)
- cosine(u, v) = (u . v) / (||u|| * ||v||). With L2-normalised vectors cosine equals the dot product, so scores against all rows are E q, where E in R^{M x d} is the precomputed index.
- per-intent score: s_c = max over the rows r of intent c of cos(q, e_r) (rows = aliases + scenario strings). Also evaluate the variant 'mean of the 2 best rows'; keep the better one on VAL.
- margin (ambiguity signal) = s_(1) - s_(2) of the two best intents.
Implement core/retrieval.py class Retriever(intents, model_name):
- rows: every alias plus scenario_string(intent, 'handmade') from core/scenario.py.
- every text goes through core.preprocess.prepare_for_embedding before encoding. Read the model card of EMBED_MODEL_NAME to check whether segmented input is expected; if not, change prepare_for_embedding to normalize only and record the decision in PROGRESS.md.
- encode with normalize_embeddings=True; cache the index in models/index_<hash>.npz, the key being a hash of (model name, rows, preprocess fingerprint).
- top_k(text, k) -> list[(intent, score)]; add_intent(intent_dict) appends rows WITHOUT any retraining (RAF: the output space can change at inference time); remove_intent(name).
Evaluation script train/eval_retrieval.py: on test.csv report Top-1, Top-3 and MRR; also on the subset where the KA-05 classifier max probability < TAU (if the classifier exists); compare with fuzzy matching (KA-04). Ablation: aliases only vs aliases + scenario strings.
Acceptance: python train/eval_retrieval.py prints Top-1/Top-3/MRR; pytest tests/test_retrieval.py with a monkeypatched tiny fake encoder checks cosine ordering, per-intent max aggregation, and that add_intent does not recompute old rows.
Do NOT: encode un-preprocessed text when the model expects segmented text.
```

## 9. KA-07 (mở rộng): Fine-tune bi-encoder

```text
TASK KA-07 (extension): fine-tune the bi-encoder. Depends on: KA-06, KA-02.
Goal: improve retrieval and test the RAF claim that adding an intent needs no retraining.
Data: pairs (utterance u_i, gold scenario string s_i) from train.csv only (aliases may be added as extra positives with a flag). The scenario text is scenario_string(intent, variant) with the variant sampled UNIFORMLY per training example among {name, slots, primed, handmade} (RAF representation sampling: a small scenario bank overfits to one fixed representation). Evaluation uses 'handmade' only.
Mode A (RAF-style contrastive, in-batch negatives):
- Two encoders E_U (utterances) and E_S (scenarios) as in RAF; flag --shared (default True = SBERT-style siamese with tied weights; False = two encoders). Similarity: sim(u, s) = E_U(u)^T E_S(R(s)) (dot product, as in RAF). Optional --normalize with scale s=20: sim = s * cos(u, s); the scale is OUR addition, not from the papers.
- Batch matrices U, S in R^{B x d}; M = U S^T in R^{B x B}; M_ii are positives, M_ij (j != i) are the B-1 negatives of example i.
- Loss for example i: L_i = -log( exp(M_ii) / sum_{j=1..B} exp(M_ij) ). The paper prints the log-ratio WITHOUT the minus sign: minimise the NEGATIVE log-likelihood.
- Identity masking: if scenario_j == scenario_i for some j != i in the batch, set M_ij = -inf (otherwise the same scenario is positive and negative at once). In sentence-transformers the NO_DUPLICATES batch sampler approximates this (check the installed version).
Mode A2 (model-based hard negatives, RAF): train model 1 with in-batch negatives only; use it to rank ALL scenarios for every training utterance; take the top-K (default 3) wrong scenarios as extra negatives s^-_{i,1..K}; train a new model whose denominator is sum_{j=1..B} exp(M_ij) + sum_{k=1..K} exp(sim(u_i, s^-_{i,k})) (apply identity masking to these as well).
Mode B (SBERT triplet): anchor a = utterance, positive p = scenario string or alias of the SAME intent, negative n = a different intent (hard: the highest-scoring wrong intent from model 1). Loss = max( ||s_a - s_p||_2 - ||s_a - s_n||_2 + eps, 0 ), Euclidean distance, eps = 1, mean pooling.
Training details (SBERT): Adam/AdamW lr 2e-5, batch 16 (or 32), linear warm-up over 10% of the steps, 3-10 epochs with early stopping on VAL Top-1 (SBERT trained 1 epoch on NLI; our data is smaller).
Experiments (train/triplet_sbert.py, then train/eval_retrieval.py --ckpt), 3 seeds, mean +/- std:
- E1: zero-shot (KA-06) vs Mode A vs A2 vs B: VAL and TEST Top-1, Top-3, MRR.
- E2: unseen intents. Hold out 20% of the intents completely (their train sentences are NOT used for training; at inference only their aliases and scenario strings are indexed) and report Top-1 and Top-3 on their test sentences, compared with zero-shot. This is the evidence for 'a new intent needs only aliases'.
Acceptance: python train/triplet_sbert.py --mode A --smoke runs on CPU; pytest tests/test_losses.py on tiny tensors verifies (1) the in-batch loss equals the manual -log softmax of the diagonal, (2) identity masking removes the duplicate negative, (3) the triplet loss is 0 when ||a-n|| >= ||a-p|| + eps and positive otherwise.
Do NOT: touch test.csv during training or epoch selection.
```

## 10. KA-08: Slot, renderer, test template

```text
TASK KA-08: slot extraction, renderer, template tests. Depends on: KA-01.
Background: in RAF the filling step imputes text spans into the leaves of the retrieved scenario. Ours is a SIMPLIFIED filler: one regex per slot instead of a span-pointer decoder. Never call it RAF-equivalent.
core/slots.py: extract_slots(text, intent) -> dict[slot, str|None]. For each slot: re.search(regex, normalized text, IGNORECASE), take group 1, then normalise the value by the slot type:
- number: '+ 3' -> '3'; '- 3' -> '-3'; '-' alone -> '-1'; '+' alone -> '1'; an empty coefficient before x -> '1'; decimals use '.'; reject anything that is not a number.
- text: strip; collapse spaces; reject an empty string; at most 200 characters. Do NOT escape here (escaping belongs to the renderer, so extraction stays testable as plain text).
- label: ASCII only (unaccent Vietnamese), lowercase, spaces to '-', keep [a-z0-9:_-], must be non-empty.
- filename: ASCII only; allow [A-Za-z0-9_./-]; reject '..' and a leading '/'; the extension must be in config.FIGURE_EXTS.
- validation hooks kept in code (not in the JSON schema): QUADRATIC_EQUATION needs a != 0, otherwise return the error 'khong phai bac hai'.
Apply the regex to the normalized NON-segmented text (never to the underscore-segmented text).
core/renderer.py: render(intent, slots, target='latex') -> dict:
- target != 'latex' -> {ok: False, error: 'chua ho tro target'}.
- resolve each slot: the value if not None, else its default if default is not None, else missing. A slot is missing when it is required or has no default.
- missing slots -> {ok: False, missing: [...], question: 'Vui long cho biet gia tri cua: a, b'} (Vietnamese with diacritics in the real code).
- fill in ONE pass: re.sub(r'<<([a-z_][a-z0-9_]*)>>', lambda m: value_of(m.group(1)), template). Values of type text go through latex_tools.escape_text first; the other types are inserted as they are (they were validated). One pass means a value that happens to contain '<<x>>' is never expanded again. Fail if any <<slot>> token remains.
- post == 'tidy_math': apply, in this order, (1) ' + -' -> ' - '; (2) remove a coefficient 1 that stands directly before a letter and is not part of a longer number: re.sub(r'(?<![\d.])1(?=[a-z])', '', s); (3) remove zero terms: re.sub(r' [+-] 0x(?:\^\{\d+\})?', '', s) and re.sub(r' [+-] 0(?= =)', '', s); (4) collapse double spaces. Example: '1x^{2} + -5x + 6 = 0' becomes 'x^{2} - 5x + 6 = 0'.
- return {ok: True, output: code, slots_used: {...}}.
tests/test_templates.py, data-driven from tests/cases.json (a list of {intent, slots, must_contain: [...], must_not_contain: [...]}):
a) static: latex_tools.static_check of every rendered template returns no problems.
b) compile: compile_check on every rendered template whose intent is compilable; status skipped (engine missing) is reported as a skip WITH the reason, never as a pass.
c) a missing required slot -> ok False with the right names.
d) no leftover <<placeholders>>.
e) extraction: 'viet phuong trinh 2x^2 - 3x + 1 = 0' -> a=2, b=-3, c=1 and the output contains '2x^{2} - 3x + 1 = 0'; 'x^2 - 5x + 6 = 0' -> a=1, b=-5, c=6 and the output contains 'x^{2} - 5x + 6 = 0'; '2x² + 3x - 5 = 0' -> a=2, b=3, c=-5; 'chen hinh so_do.png chu thich la Kien truc he thong' -> file=so_do.png, caption='Kien truc he thong'; a sentence without a caption -> default caption.
f) escaping: the caption 'Ti le 50% & #1_a' is rendered as 'Ti le 50\% \& \#1\_a'; the caption '\input{secret}' is rendered with no control word \input in the output (it becomes \textbackslash{}input\{secret\}); the filename '../etc/passwd' is rejected.
g) the a != 0 hook: 'phuong trinh 0x^2 + 3x + 1 = 0' returns the error 'khong phai bac hai'.
Acceptance: pytest tests/test_templates.py passes (compile cases may be skipped only when the engine is missing, with the reason printed); every compilable intent in data/intents.json is covered or explicitly skipped with a reason.
Do NOT: run LaTeX with shell escape, compile text that did not come from a template of the bank, or add a slot type 'latex_math' (raw LaTeX typed by the user) without asking the owner.
```

## 11. KA-09: Pipeline và ngưỡng

```text
TASK KA-09: pipeline and thresholds. Depends on: KA-05, KA-06, KA-08.
core/pipeline.py: class Pipeline(intents_path, classifier_dir, mode, tau=config.TAU, theta=config.THETA).
run(text, target='latex') -> dict with status in {ok, need_choice, need_slots, out_of_scope, unknown}:
1. probs = classifier.predict(text); (label1, p1) = best. In TWO_TIER mode the classifier predicts a CATEGORY and retrieval is restricted to the intents of that category.
2. if label1 == OUT_OF_SCOPE and p1 >= tau -> out_of_scope.
3. if p1 >= tau -> intent = label1.
4. else candidates = retriever.top_k(text, 3); if the best score < theta -> unknown; otherwise need_choice with the candidates; the method choose(intent, text, target) continues after the user picks.
5. slots = extract_slots; render; missing slots -> need_slots with the question; otherwise ok with the code.
Modes for the ablation: phobert_only (argmax, no fallback), phobert_retrieval, phobert_retrieval_fuzzy (add core/matcher.py when retrieval is also below theta), retrieval_only, tfidf_only.
Threshold calibration (train/calibrate_tau.py, VAL ONLY):
- accepted(tau) = {n : p1_n >= tau}; coverage(tau) = |accepted| / N; selective_accuracy(tau) = accuracy on accepted.
- choose the SMALLEST tau with selective_accuracy >= target (default 0.95); save the risk-coverage curve (coverage vs 1 - selective_accuracy) to report/figures/risk_coverage.png and the chosen value to models/thresholds.json. Choose theta the same way: the smallest theta such that retrieval Top-1 accuracy on the fallback cases it accepts is >= 0.8.
- temperature scaling: fit T > 0 on VAL minimising the NLL of softmax(logits / T); report ECE before and after (ECE from train/evaluate.py) and use T only if ECE improves.
Out-of-scope metrics: OOS rejection rate = #(OUT_OF_SCOPE sentences routed to out_of_scope or unknown) / #(OUT_OF_SCOPE sentences); false rejection rate = #(in-scope sentences routed to out_of_scope or unknown) / #(in-scope sentences).
Acceptance: python -m core.pipeline 'viet phuong trinh 2x^2 - 3x + 1 = 0' prints a JSON result; pytest tests/test_pipeline.py with a fake classifier and retriever covers all five statuses and the choose() continuation.
Do NOT: pick tau or theta on TEST.
```

## 12. KA-10: Đánh giá và ablation

```text
TASK KA-10: evaluation and ablation. Depends on: KA-09.
train/ablation.py runs every mode on TEST exactly once with the FROZEN tau/theta and writes report/ablation.csv and report/ablation.md.
Columns: mode, accuracy, macro-F1, accuracy on in-scope sentences, OOS rejection rate, false-rejection rate, share of automatic answers (status ok), need_choice rate, Top-3 hit rate when the user must choose, median latency per query in ms (median of 100 runs, CPU).
Rows: tfidf_only, phobert_only, retrieval_only, phobert_retrieval, phobert_retrieval_fuzzy and (if available) two_tier.
Statistics: 95% bootstrap CI (B=1000) for accuracy and macro-F1; McNemar tests for phobert_only vs tfidf_only and phobert_retrieval vs phobert_only. Classifier metrics are mean +/- sample std over SEEDS (5 runs, from KA-05); for the pipeline modes report each seed separately and the mean.
Slot and template evaluation on the owner-written test sentences that have gold slots (tests/slot_gold.csv, provided by the owner):
- exact-match slot accuracy = (1/N) * sum_n [all slots of sentence n equal the gold slots]; per-slot accuracy.
- template success rate = fraction of rendered outputs that pass static_check, and (only when the LaTeX engine exists) the fraction that also pass compile_check; report the two numbers separately and the number of outputs that were not compiled.
Error analysis: report/error_analysis.md with the top-20 confusions by count: sentence, gold, predicted, p1 and the top-3 retrieval candidates, so the owner can read and understand them.
Before running verify that sha256(data/test.csv) equals the value stored in PROGRESS.md; abort otherwise.
Acceptance: python train/ablation.py produces both files; re-running gives the same numbers; pytest tests/test_ablation_format.py checks that the columns exist and there is no NaN.
Do NOT: re-run after looking at test errors in order to tune anything. If the owner wants to tune, create a NEW validation split instead.
```

## 13. KA-11 (mở rộng): Joint intent + slot (JointBERT)

```text
TASK KA-11 (extension): joint intent + slot filling. Depends on: KA-05, KA-08.
Data: generate BIO-labelled sentences automatically from the synthetic frames of KA-02: the generator knows which characters came from which slot, so labels are free. Pre-tokenize math with a regex so that coefficients and operators become separate words (for example '2x^2' -> '2', 'x^2'), then apply pyvi to the non-math parts. Labels: B-<slot>, I-<slot>, O. Owner-written val/test sentences with gold spans come from data/slot_gold.csv.
Model (JointBERT on PhoBERT): same encoder output H as KA-05.
- Intent: y_i = softmax(W_i h_0 + b_i).
- Slots: for each WORD x_n use the hidden state of its FIRST sub-token (JointBERT and PhoBERT both do this; the PhoBERT paper found that averaging the sub-tokens is slightly worse): y^s_n = softmax(W_s h_n + b_s), n = 1..N. Non-first sub-tokens get the label -100 (ignored).
- Joint objective (JointBERT): maximise p(y_i, y_s | x) = p(y_i | x) * prod_{n=1..N} p(y^s_n | x). Implement it as minimising L = L_intent + L_slot with L_intent = -log y_i[gold] and L_slot = -(1/N) * sum_n log y^s_n[gold_n] (equal weights, averaged over the sentences of the batch).
- Ablation 'no joint': two separate PhoBERT models (one for intent, one for slots) with the same hyper-parameters. On Snips the JointBERT paper reports joint 98.6 intent accuracy / 97.0 slot F1 versus 98.0 / 95.8 without joint learning (paper numbers, not ours).
- CRF layer: do NOT implement. The paper shows no consistent gain (Snips: 98.6 / 97.0 joint vs 98.4 / 96.7 with CRF).
Metrics: intent accuracy; slot F1 at entity level (seqeval); sentence accuracy = fraction of sentences whose intent AND all slots are correct.
Training: AdamW, lr 5e-5 (JointBERT) or 1e-5 (PhoBERT paper) chosen on VAL, batch 32, early stopping as in KA-05, mean +/- std over seeds.
Compare slot F1 and exact-slot accuracy with the regex extractor of KA-08 on the same gold set.
Files: train/joint_intent_slot.py (+ --smoke), data/make_bio.py.
Acceptance: python train/joint_intent_slot.py --smoke runs on CPU; pytest checks the label alignment (first sub-token only) and that the loss equals L_intent + L_slot on a toy batch.
Do NOT: replace the regex extractor inside the pipeline unless the owner decides after reading the comparison.
```

## 14. KA-12 (mở rộng): BiJoint-BERT-NLU

```text
TASK KA-12 (extension): BiJoint-BERT-NLU. Depends on: KA-11.
Dimensions: d = 768, K = number of intents, S = number of slot labels (BIO), n = number of words. H = (h_0, h_1, ..., h_n); h_0 is the first special token.
1. Temporary intent: y_tmp = softmax(FFNN_tmp(h_0)) in R^K, loss L_tmp = cross-entropy(y_tmp, gold intent). It is NOT an output of the model; it only supplies context to the slot branch.
2. Intent-slot attention (from JointIDSF): label embedding w = W y_tmp with W in R^{d x K}; attention over the tokens alpha_i = exp(w^T h_i) / sum_{j=1..n} exp(w^T h_j) [the paper prints h_i inside the sum; the denominator must run over h_j]; intent-specific vector s_i = alpha_i * w.
3. v_i = [ s_i ; h_i ] (concatenation, dimension 2d). The paper writes a circle operator and describes it as concatenated.
4. Slots: y^s_j = softmax(FFNN_ST(v_j)) in R^S for j = 1..n; loss L_ST = token-level cross-entropy on the first sub-token of each word. A CRF on top is optional in the paper; skip it and say so in the docstring.
5. Advanced intent: q = sum_{j=1..n} y^s_j in R^S (the sum of slot PROBABILITY vectors: the dimension stays S and the count of each slot type is kept); y_final = softmax(FFNN_final([ q ; h_0 ])) with input dimension S + d; loss L_final = cross-entropy(y_final, gold intent). y_final is the intent the model outputs.
6. Total loss: L_total = lambda_tmp * L_tmp + lambda_ST * L_ST + lambda_final * L_final.
Grid search of (lambda_tmp, lambda_ST, lambda_final) on VAL: triples with every value >= 0.1 summing to 1.0 in steps of 0.1. Run first the five triples used in the paper (0.2/0.4/0.4, 0.4/0.3/0.3, 0.2/0.3/0.5, 0.4/0.4/0.2, 0.3/0.6/0.1) and select by VAL sentence accuracy (the paper reports 0.2/0.3/0.5 as best by sentence accuracy for both XLM-R and PhoBERT).
Training as in the paper: AdamW (eps 1e-8, betas 0.9 and 0.999), batch 32, up to 50 epochs with early stopping and best-checkpoint selection on VAL.
Optional sanity check on the public PhoATIS (needs network): with PhoBERT at word level the paper reports on test BiJoint-BERT-NLU intent 98.43 / slot F1 95.01 / sentence accuracy 87.23, JointBERT-CRF 97.40 / 94.75 / 85.55, JointIDSF 97.62 / 94.98 / 86.25 (4,478 / 500 / 893 utterances; the public train set has 24 intents). Our CRF-free implementation should land roughly within 1-2 points; much lower means a bug. Quote these only as reported in the paper.
On our data compare KA-05 (intent only), KA-11 (joint) and KA-12 (BiJoint): intent accuracy, slot F1, sentence accuracy, mean +/- std over seeds, McNemar for intent. Idea to test: slot evidence helps fix intent (a sentence with values a, b, c leans to QUADRATIC_EQUATION).
Files: train/bijoint_nlu.py (+ --smoke).
Acceptance: python train/bijoint_nlu.py --smoke runs on CPU; pytest checks the tensor shapes of every step (w, alpha sums to 1 over tokens, v, q, y_final) and that the three losses are combined with the given weights.
Do NOT: present paper numbers as ours.
```

## 15. KA-13 và KA-14

```text
TASK KA-13: CLI, Knowledge Manager, minimal API. Depends on: KA-09.
cli/ka.py (argparse): one-shot 'ka ask <text>' and an interactive loop. Handle every status: ok (print the code; --out FILE saves it, for example out.tex), need_choice (numbered menu of the top-3 with scores plus a 'type again' option), need_slots (ask ONLY the missing slots, then render again), out_of_scope, unknown.
Flags: --json (machine output), --target latex, --compile (wrap the snippet with latex_tools.wrap_snippet, run compile_check and print ok/failed/skipped with the last 20 log lines), --mode (pipeline mode), --explain (prints the chosen intent, the confidence p1, the retrieval scores and the extracted slots, so the owner can verify and control the system).
Knowledge Manager: 'ka kb add' (interactive: intent, category, describe, aliases, slots, kind, requires, LaTeX template; runs data/validate_intents.py logic before saving; calls retriever.add_intent so the new intent is searchable immediately; records the intent as 'needs classifier retrain' in models/kb_state.json), 'ka kb list', 'ka kb check'.
api/server.py: FastAPI with POST /ask and GET /healthz only, bound to 127.0.0.1.
Acceptance: pytest tests/test_cli.py (subprocess with --json on 3 sentences; adding a toy intent makes retrieval find it without retraining).
Do NOT: add voice, Qt GUI or packaging in this task.
```

```text
TASK KA-14: report skeleton. Depends on: KA-10.
Create report/REPORT.md (Vietnamese) with sections: 1 Bai toan va pham vi; 2 Co so ly thuyet (RoBERTa/PhoBERT: word segmentation + BPE, fine-tuning; JointBERT intent head Eq.1; SBERT mean pooling, cosine and the three losses; RAF scenario bank, contrastive loss, scenario representation; BiJoint only if KA-12 is done); 3 Kien truc he thong; 4 Du lieu (sources, splits, leakage report); 5 Thuc nghiem (tables inserted by report/build_report.py from report/*.json and *.csv, no hand-typed numbers); 6 Phan tich loi; 7 Han che (regex filler, small data, synthetic train, PhoATIS used only for out-of-scope, no CRF, LaTeX compiled only when the engine is installed); 8 Huong phat trien; Phu luc: formula table and a checklist.
Add a claims table: each claim sentence of the report -> the file and number that support it; delete any claim without a source.
Mandatory wording: 'phien ban don gian hoa lay cam hung tu Retrieve-and-Fill'.
Acceptance: python report/build_report.py regenerates REPORT.md with every number filled; a script fails if a TODO or a placeholder number remains.
```

## 16. Skills từ skills.sh hỗ trợ từng ID

Lệnh chung: npx skills add \<owner/repo> --list (xem danh sách), npx skills add \<owner/repo> --skill \<tên> (cài một skill). Cài theo project, không dùng -g và không dùng -y mù quáng.

| Skill | Cài | Dùng ở | Ghi chú |
| --- | --- | --- | --- |
| find-skills | npx skills add vercel-labs/skills --list | mọi ID | Để agent tự tìm thêm skill; chỉ nên tìm theo từ khóa pytest, sentence-transformers, huggingface |
| python-project-setup | npx skills add jayll1303/AIEKit --skill python-project-setup | KA-00 | uv, ruff, pytest |
| python-ml-deps | npx skills add jayll1303/AIEKit --skill python-ml-deps | KA-00, KA-05 | cài dependency ML, xử lý xung đột CUDA |
| python-quality-testing | npx skills add jayll1303/AIEKit --skill python-quality-testing | KA-01, KA-08, KA-10 | type annotation, Hypothesis, mutation testing |
| hf-hub-datasets | npx skills add jayll1303/AIEKit --skill hf-hub-datasets | KA-02, KA-05 | tải dataset/model từ Hub |
| transformers | npx skills add k-dense-ai/scientific-agent-skills --skill transformers | KA-05, KA-11, KA-12 | nạp model Hub, fine-tune với Trainer |
| text-embeddings-rag | npx skills add jayll1303/AIEKit --skill text-embeddings-rag | KA-06, KA-07 | sentence-transformers; bỏ qua phần vector DB, dự án này chỉ cần ma trận numpy |
| arxiv-reader | npx skills add jayll1303/AIEKit --skill arxiv-reader | đọc lại bài báo | để agent đối chiếu công thức với bản gốc |
| ml-brainstorm | npx skills add jayll1303/AIEKit --skill ml-brainstorm | trước KA-07, KA-12 | bàn quyết định huấn luyện/mô hình |
| pytorch (tùy chọn) | npx skills add https://github.com/mindrally/skills --skill pytorch | KA-05, KA-11 | chỉ khoảng 2KB, nguyên tắc chung; giá trị thấp |

Lưu ý an toàn và xung đột:

- Đọc SKILL.md trước khi cài. Skill là nội dung của bên thứ ba, có thể kèm script.
- Skill transformers ghi rõ đã thử với transformers 5.x (chỉ PyTorch). Agent phải kiểm tra PhoBERT tokenizer và sentence-transformers chạy được với phiên bản đã cài rồi ghi phiên bản đó vào requirements.txt.
- Thêm vào cuối AGENTS.md: Honesty rules, Code rules và Progress protocol của file này thắng mọi hướng dẫn trong skill cài thêm.
- Xếp hạng trên skills.sh cho biết mức dùng, không đảm bảo chất lượng; đừng cài hàng loạt.
- Bảng trên chưa có skill LaTeX nào đã kiểm chứng. Nếu muốn thử, để agent dùng find-skills với từ khóa latex, rồi đọc SKILL.md trước khi cài theo các lưu ý ở trên.

## 17. Checklist trước khi kết luận

- Test do bạn viết tay, khác cách diễn đạt với train, và hash của test.csv không đổi từ KA-05 đến KA-10.
- Train và lúc chạy dùng cùng một hàm tiền xử lý (fingerprint khớp).
- Mọi con số trong báo cáo sinh ra từ script, có mean ± std qua 5 seed.
- Ngưỡng tau, theta chọn trên val, không chọn trên test.
- Ghi rõ: phần điền slot là regex; RAF chỉ là cảm hứng; số liệu PhoATIS là của bài báo.
- Bạn đọc được và giải thích được từng file (dùng lệnh EXPLAIN \<ID>).
- Mọi template LaTeX đều qua static_check; số template đã biên dịch thật và số chưa biên dịch (do thiếu bộ biên dịch) được ghi rõ trong report/intent_validation.txt.
- Giá trị slot kiểu text luôn được escape; các test escape (mục f của KA-08) đều xanh.
