# Vietnamese LaTeX Knowledge Assistant

Vietnamese LaTeX Knowledge Assistant is a scenario-based NLP assistant designed to process natural language Vietnamese math and STEM queries and generate formatted LaTeX code. The system architecture is inspired by Retrieve-and-Fill (RAF), employing PhoBERT for intent classification, Sentence-BERT for semantic retrieval fallback, and regular-expression-based slot extraction.

---

## Architecture Overview

```
User Query (Vietnamese text)
  │
  ▼
Preprocess (Unicode NFC normalization, math symbol normalization, PyVi word segmentation)
  │
  ▼
PhoBERT Intent Classifier (Softmax probability p over intents)
  ├─ If max(p) >= tau (0.8): Intent accepted
  └─ If max(p) < tau: Semantic Retrieval Fallback (Sentence-BERT top-3 over scenario aliases)
       ├─ If top similarity >= theta (0.3): Present top-3 choices to user
       └─ If top similarity < theta (0.3): Reject query (intent not understood)
  │
  ▼
Slot Extraction (Regular expressions per intent)
  ├─ Missing required slots: Prompt user for missing parameters
  └─ All required slots present: Proceed to rendering
  │
  ▼
LaTeX Template Rendering (Text slots escaped, safe compilation with -no-shell-escape)
  │
  ▼
Rendered Output
```

> **Note on Methodology**: Slot filling uses deterministic regular expressions rather than complex span-pointer decoders. The system is designed as a simplified scenario-based framework inspired by Retrieve-and-Fill (RAF).

---

## Environment Setup

Set up a standard Python virtual environment using the three-command workflow below:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> On Windows PowerShell, activate the virtual environment with:
> ```powershell
> .venv\Scripts\Activate.ps1
> ```

---

## Dependencies

Project dependencies in `requirements.txt` are categorized as follows:

- **PyTorch**: `torch` for tensor computations and neural network inference.
- **Transformers**: `transformers` for PhoBERT model loading and inference, alongside `pyvi` for Vietnamese word segmentation.
- **Sentence-Transformers**: `sentence-transformers` for embedding generation and bi-encoder cosine similarity retrieval fallback.
- **Web API**: `fastapi` and `uvicorn` for serving API endpoints.
- **Evaluation**: `scikit-learn`, `numpy`, `pandas`, and `evaluate` for benchmark scoring, metrics calculation, and dataset handling.
- **Sequence tagging metrics**: `seqeval` for entity-level BIO slot F1.
- **Code Quality**: `pytest` for unit testing, `ruff` for linting and code formatting, and `mypy` for static type checking.

---

## Testing and Verification

Run the test suite using pytest:

```bash
python3 -m pytest
```

Check code quality with ruff:

```bash
ruff check .
```

---

## Joint intent-slot training

Write owner-authored validation and test spans in `data/slot_gold.csv` using the schema in [data/SLOT_GOLD_SCHEMA.md](data/SLOT_GOLD_SCHEMA.md). Generate BIO labels from the KA-02 frames and then run the experiments:

```bash
python data/make_bio.py --synthetic-train
python train/joint_intent_slot.py --data data/slot_gold_bio.csv
python train/bijoint_nlu.py --data data/slot_gold_bio.csv
python report/build_report.py
```

The JointBERT runner tunes learning rate on validation, then trains five seeds, the KA-05 intent-only baseline and the separate-model no-joint ablation. BiJoint runs the five paper weight triples first; pass `--full-grid` to evaluate every valid tenth-step triple. Both save checkpoints under `models/checkpoints/` and result JSON in `report/`. They require PhoBERT base weights to be available from Hugging Face or the local model cache. The repository does not include owner gold spans.

CPU smoke checks that do not download model weights:

```bash
python train/joint_intent_slot.py --smoke
python train/bijoint_nlu.py --smoke
```

## CLI and local API

```bash
python cli/ka.py ask "tạo phân số tử 2 mẫu 3" --explain
python cli/ka.py ask --json "viết phương trình bậc hai"
python cli/ka.py kb list
python cli/ka.py kb check
python api/server.py
```

The API listens on `127.0.0.1:8000` and exposes `POST /ask` and `GET /healthz`. Open `http://127.0.0.1:8000/ui/` for the local Knowledge Manager dashboard and Contract v2 definition wizard. The wizard uses the management API for definitions and keeps the corpus catalog read-only and separate. See [docs/KNOWLEDGE_MANAGER.md](docs/KNOWLEDGE_MANAGER.md) for preview, import/export, publish and rollback steps. `--offline-mock` is only for deterministic CLI tests; it does not represent semantic retrieval quality.
## Intent targets and slot schema

Target availability, `absent_means`, and registered `post` actions are driven by
`data/intents.json`; see [docs/INTENT_SCHEMA.md](docs/INTENT_SCHEMA.md). A selected
intent without a template for the requested target returns `unsupported_target`.
LaTeX may fall back to MathJax. Spoken math normalization applies only to
number/math slots; unsupported forms stay unfilled for clarification.
