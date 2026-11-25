## Project
Local Vietnamese knowledge assistant. The user types a Vietnamese sentence. The system (1) picks a scenario (intent + template) from a user-managed scenario bank (data/intents.json), (2) extracts slot values, (3) fills a LaTeX template and returns LaTeX code. Only the latex template is in scope now; keep the keys python and cpp in the schema but leave them empty.
Academic goal: study and evaluate ONE model: the RoBERTa family for Text Classification (project #5 of 11). PhoBERT (pre-training approach based on RoBERTa) is the Vietnamese encoder used for intent classification. Sentence-BERT-style retrieval is the fallback and handles NEW intents without retraining. Slot filling in the core pipeline is regex-based.
## Environment
Linux (Gentoo), Python 3.11+, CPU must work, GPU optional. Cache models under models/hf_cache. Do not assume internet after dependencies and models are downloaded. LaTeX compile tests need pdflatex (TeX Live); it is optional: when it is missing, skip the compile tests with an explicit reason and never install it.

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