"""Quản lý tự học thực tế (learned_aliases) và huấn luyện/đánh giá theo mốc 10 kiến thức mới.

Thiết kế tách biệt hoàn toàn với tập benchmark học thuật data/test.csv (Honesty Rule #3):
- Mọi dữ liệu feedback, user_train/val và báo cáo mốc được lưu riêng dưới thư mục runtime
  cạnh intents.json đang chạy.
- Từ kiến thức mới thứ 1..9: Retrieval + Lexical Boost hoạt động tức thì (<0.1s).
- Đạt mốc thứ 10 (MILESTONE_TRAIN_THRESHOLD): tự động hoặc theo lệnh sinh tập train/val
  cho kho hiện hành, đo Val Top-1 trước & sau, fine-tune 2 tầng trên Bi-Encoder và chỉ cập
  nhật trọng số nếu chất lượng không giảm.
"""

from __future__ import annotations

import copy
import csv
import json
import random
import shutil
import threading
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import (
    DEFAULT_SEED,
    MILESTONE_TRAIN_THRESHOLD,
    OUT_OF_SCOPE_INTENT,
    SBERT_DIR,
    SBERT_MODEL_NAME,
    get_default_device,
    resolve_hf_snapshot,
    set_seed,
)
from core.preprocess import prepare_for_embedding, preprocess_fingerprint
from core.scenario import scenario_string

_TRAIN_OPENERS = (
    "hướng dẫn",
    "cách làm",
    "cho tôi ví dụ về",
    "tạo giúp tôi",
    "tra cứu nhanh",
    "cú pháp cho",
)
_VAL_OPENERS = (
    "bạn chỉ mình",
    "mình cần xem mẫu",
    "làm sao thực hiện",
    "tìm nhanh kiến thức về",
)


def _norm_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value or "")).strip().lower().split())


def _is_user_custom_definition(intent_id: str, definition: dict[str, Any]) -> bool:
    if intent_id == OUT_OF_SCOPE_INTENT:
        return False
    return str(definition.get("domain", "")).strip() == "user-created"


class KnowledgeLearningManager:
    """Theo dõi feedback truy vấn, tiến độ mốc 10 kiến thức mới và chạy fine-tune nền an toàn."""

    def __init__(self, intents_path: Path | str, model_dir: Path | str | None = None) -> None:
        self.intents_path = Path(intents_path)
        self.learning_dir = self.intents_path.parent / "learning"
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.learning_dir / "milestone_state.json"
        self.feedback_path = self.learning_dir / "user_feedback.jsonl"
        self.train_csv_path = self.learning_dir / "milestone_train.csv"
        self.val_csv_path = self.learning_dir / "milestone_val.csv"
        self.report_path = self.learning_dir / "milestone_report.json"
        self.model_dir = Path(model_dir) if model_dir is not None else Path(SBERT_DIR)
        self._lock = threading.RLock()
        self._running = False
        self._last_error = ""

    def _read_state(self) -> dict[str, Any]:
        default = {
            "threshold": MILESTONE_TRAIN_THRESHOLD,
            "pending_intents": [],
            "trained_intents": [],
            "last_trained_at": None,
            "last_report": None,
            "feedback_count": 0,
        }
        if not self.state_path.is_file():
            return default
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {**default, **data}
        except (OSError, json.JSONDecodeError):
            pass
        return default

    def _write_state(self, state: dict[str, Any]) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.state_path)

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = self._read_state()
            pending = list(state.get("pending_intents", []))
            return {
                "threshold": MILESTONE_TRAIN_THRESHOLD,
                "pending_count": len(pending),
                "pending_intents": pending,
                "milestone_ready": len(pending) >= MILESTONE_TRAIN_THRESHOLD,
                "training_running": self._running,
                "last_error": self._last_error,
                "last_trained_at": state.get("last_trained_at"),
                "last_report": state.get("last_report"),
                "feedback_count": int(state.get("feedback_count", 0)),
                "learning_dir": str(self.learning_dir),
            }

    def mark_published(
        self,
        snapshot: dict[str, dict[str, Any]],
        changed_intents: list[str] | tuple[str, ...],
    ) -> dict[str, Any]:
        """Ghi nhận kiến thức mới/chỉnh sửa vào hàng đợi mốc 10."""
        with self._lock:
            state = self._read_state()
            pending = list(state.get("pending_intents", []))
            trained = set(state.get("trained_intents", []))
            for intent in changed_intents:
                if intent not in snapshot:
                    continue
                if intent not in pending and (
                    _is_user_custom_definition(intent, snapshot[intent]) or intent not in trained
                ):
                    pending.append(intent)
            # Bỏ intent đã bị xóa
            pending = [intent for intent in pending if intent in snapshot]
            state["pending_intents"] = pending
            self._write_state(state)
            return self.status()

    def record_feedback(
        self,
        *,
        query: str,
        intent: str,
        target: str,
        accepted: bool = True,
    ) -> dict[str, Any]:
        """Lưu cặp (câu hỏi thật, intent đã chọn) vào nhật ký học liệu thực tế."""
        cleaned_query = " ".join(str(query or "").split()).strip()
        if not cleaned_query or not intent:
            return {"recorded": False}
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": cleaned_query,
            "intent": intent,
            "target": target,
            "accepted": accepted,
        }
        with self._lock:
            with self.feedback_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            state = self._read_state()
            state["feedback_count"] = int(state.get("feedback_count", 0)) + 1
            self._write_state(state)
            return {"recorded": True, **entry}

    def build_milestone_datasets(
        self,
        snapshot: dict[str, dict[str, Any]],
        *,
        seed: int = DEFAULT_SEED,
    ) -> dict[str, Any]:
        """Sinh tập user train/val độc lập (không trùng lặp, không đụng data/test.csv)."""
        rng = random.Random(seed)
        feedback_by_intent: dict[str, list[str]] = {}
        if self.feedback_path.is_file():
            for line in self.feedback_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("accepted") and rec.get("query") and rec.get("intent"):
                        feedback_by_intent.setdefault(str(rec["intent"]), []).append(str(rec["query"]))
                except json.JSONDecodeError:
                    continue

        state = self._read_state()
        priority_intents = set(state.get("pending_intents", []))
        all_candidates = [k for k in snapshot if k != OUT_OF_SCOPE_INTENT]
        # Ưu tiên kiến thức mới/tùy biến + mẫu đại diện kiến thức nền (tránh quên kiến thức cũ)
        user_created = [k for k in all_candidates if _is_user_custom_definition(k, snapshot[k]) or k in priority_intents]
        base_intents = [k for k in all_candidates if k not in user_created]
        rng.shuffle(base_intents)
        selected_intents = user_created + base_intents[: max(30, min(len(base_intents), len(user_created) * 4 + 30))]

        seen_norm: set[str] = set()
        train_rows: list[dict[str, str]] = []
        val_rows: list[dict[str, str]] = []

        for intent in selected_intents:
            item = snapshot[intent]
            cat = str(item.get("category", "general"))
            raw_pool: list[str] = []
            for field_name in ("learned_aliases", "aliases"):
                vals = item.get(field_name) or []
                if isinstance(vals, list):
                    raw_pool.extend(str(v).strip() for v in vals if str(v).strip())
            raw_pool.extend(feedback_by_intent.get(intent, []))
            desc = str(item.get("describe", "")).split("|")[0].strip()
            display = str(item.get("display_name", "")).strip()
            if desc:
                raw_pool.append(desc)
            if display:
                raw_pool.append(display)

            unique_raw: list[str] = []
            local_seen: set[str] = set()
            for text in raw_pool:
                nt = _norm_text(text)
                if nt and nt not in local_seen and nt not in seen_norm:
                    local_seen.add(nt)
                    unique_raw.append(text)

            expanded_train: list[str] = list(unique_raw)
            for base_txt in unique_raw[:4]:
                for op in _TRAIN_OPENERS:
                    cand = f"{op} {base_txt}".strip()
                    nt = _norm_text(cand)
                    if nt not in seen_norm and nt not in local_seen:
                        local_seen.add(nt)
                        expanded_train.append(cand)

            val_cands: list[str] = []
            for base_txt in reversed(unique_raw[:3]):
                for op in _VAL_OPENERS:
                    cand = f"{op} {base_txt}".strip()
                    nt = _norm_text(cand)
                    if nt not in seen_norm and nt not in local_seen:
                        local_seen.add(nt)
                        val_cands.append(cand)
                        if len(val_cands) >= 2:
                            break
                if len(val_cands) >= 2:
                    break

            for txt in val_cands[:2]:
                seen_norm.add(_norm_text(txt))
                val_rows.append({"text": txt, "label": intent, "category": cat})
            for txt in expanded_train[:12]:
                nt = _norm_text(txt)
                if nt not in seen_norm:
                    seen_norm.add(nt)
                    train_rows.append({"text": txt, "label": intent, "category": cat})

        for path, rows in ((self.train_csv_path, train_rows), (self.val_csv_path, val_rows)):
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["text", "label", "category"])
                writer.writeheader()
                writer.writerows(rows)

        return {
            "selected_intents": len(selected_intents),
            "user_intents": len(user_created),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "train_csv": str(self.train_csv_path),
            "val_csv": str(self.val_csv_path),
        }

    def run_milestone_training(
        self,
        snapshot: dict[str, dict[str, Any]],
        *,
        on_model_ready: Any | None = None,
        epochs: int = 2,
    ) -> dict[str, Any]:
        """Sinh tập train/val, đánh giá trước -> fine-tune top-2 layers -> đánh giá sau -> cập nhật nếu cải thiện."""
        with self._lock:
            if self._running:
                return {"started": False, "reason": "Đang có tiến trình huấn luyện mốc chạy nền."}
            self._running = True
            self._last_error = ""

        t0 = time.time()
        try:
            set_seed(DEFAULT_SEED)
            data_stats = self.build_milestone_datasets(snapshot)
            if data_stats["train_rows"] < 10 or data_stats["val_rows"] < 4:
                raise RuntimeError("Dữ liệu sinh cho mốc chưa đủ mẫu để huấn luyện và kiểm định.")

            import numpy as np
            import torch
            from sentence_transformers import SentenceTransformer
            from torch.utils.data import DataLoader, Dataset
            from train.triplet_sbert import (
                create_biencoder_model,
                create_linear_schedule_with_warmup,
                in_batch_contrastive_loss,
                load_tokenizer,
            )
            from core.retrieval import Retriever

            base_dir = self.model_dir if self.model_dir.is_dir() else resolve_hf_snapshot(SBERT_MODEL_NAME)
            if base_dir is None:
                raise RuntimeError("Không tìm thấy mô hình vietnamese-bi-encoder cục bộ.")

            device = get_default_device()
            train_rows = list(csv.DictReader(self.train_csv_path.open("r", encoding="utf-8-sig")))
            val_rows = list(csv.DictReader(self.val_csv_path.open("r", encoding="utf-8-sig")))

            def _eval_retriever(model_path: Path | str) -> dict[str, Any]:
                ret = Retriever(
                    model=str(model_path),
                    model_name=SBERT_MODEL_NAME,
                    intents=list(snapshot.values()),
                    use_cache=False,
                    device=device,
                )
                q_emb = ret.model.encode(
                    [prepare_for_embedding(r["text"]) for r in val_rows],
                    batch_size=32,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                sc = np.matmul(q_emb, ret._embeddings.T)
                row_intents = [row.intent for row in ret._rows]
                top1 = top3 = 0
                for idx, row in enumerate(val_rows):
                    seen_it: list[str] = []
                    for j in np.argsort(-sc[idx]):
                        it = row_intents[j]
                        if it not in seen_it:
                            seen_it.append(it)
                            if len(seen_it) == 3:
                                break
                    top1 += int(bool(seen_it) and seen_it[0] == row["label"])
                    top3 += int(row["label"] in seen_it)
                return {
                    "val_top1": round(top1 / len(val_rows), 4),
                    "val_recall3": round(top3 / len(val_rows), 4),
                    "total": len(val_rows),
                }

            before_metrics = _eval_retriever(base_dir)
            intent_names = sorted(k for k in snapshot if k != OUT_OF_SCOPE_INTENT)
            intent2idx = {name: i for i, name in enumerate(intent_names)}
            prep_targets = {
                name: [
                    prepare_for_embedding(str(t))
                    for t in (
                        [scenario_string(snapshot[name], variant="handmade")]
                        + list(snapshot[name].get("aliases") or [])
                    )
                ]
                for name in intent_names
            }
            prep_pairs = [
                (
                    prepare_for_embedding(r["text"]),
                    random.Random(DEFAULT_SEED + idx).choice(prep_targets[r["label"]]),
                    intent2idx[r["label"]],
                )
                for idx, r in enumerate(train_rows)
                if r["label"] in intent2idx
            ]

            class _PairDS(Dataset):
                def __len__(self) -> int:
                    return len(prep_pairs)

                def __getitem__(self, idx: int) -> tuple[str, str, int]:
                    return prep_pairs[idx]

            tokenizer = load_tokenizer(base_dir)
            model = create_biencoder_model(
                shared_weights=True,
                smoke_mode=False,
                model_name_or_path=base_dir,
                device=device,
            )
            backbone = model.query_encoder.backbone
            layers = backbone.encoder.layer
            for param in backbone.embeddings.parameters():
                param.requires_grad_(False)
            for idx, layer in enumerate(layers):
                if idx < max(0, len(layers) - 2):
                    for param in layer.parameters():
                        param.requires_grad_(False)
            trainable = [p for p in model.parameters() if p.requires_grad]

            def _collate(batch: list[tuple[str, str, int]]) -> dict[str, Any]:
                qs = [b[0] for b in batch]
                ps = [b[1] for b in batch]
                idxs = torch.tensor([b[2] for b in batch], dtype=torch.long)
                q_tok = tokenizer(qs, padding=True, truncation=True, max_length=64, return_tensors="pt")
                p_tok = tokenizer(ps, padding=True, truncation=True, max_length=64, return_tensors="pt")
                return {"queries": q_tok, "positives": p_tok, "scenario_indices": idxs}

            loader = DataLoader(_PairDS(), batch_size=min(32, max(4, len(prep_pairs))), shuffle=True, collate_fn=_collate)
            optimizer = torch.optim.AdamW(trainable, lr=2e-5, weight_decay=0.01)
            total_steps = max(1, len(loader) * epochs)
            scheduler = create_linear_schedule_with_warmup(optimizer, max(1, int(total_steps * 0.1)), total_steps)

            for _ in range(epochs):
                model.train()
                for batch in loader:
                    optimizer.zero_grad()
                    qe = model.encode_query(
                        batch["queries"]["input_ids"].to(device),
                        batch["queries"]["attention_mask"].to(device),
                    )
                    pe = model.encode_doc(
                        batch["positives"]["input_ids"].to(device),
                        batch["positives"]["attention_mask"].to(device),
                    )
                    loss = in_batch_contrastive_loss(
                        queries=qe,
                        positives=pe,
                        scenario_indices=batch["scenario_indices"].to(device),
                        metric="scaled_cosine",
                        scale=20.0,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                    optimizer.step()
                    scheduler.step()

            staging_dir = self.learning_dir / "staging_sbert"
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            st_model = SentenceTransformer(str(base_dir), device="cpu")
            st_model[0].auto_model.load_state_dict(
                copy.deepcopy(model.query_encoder.backbone.cpu().state_dict()),
                strict=True,
            )
            st_model.save(str(staging_dir))

            after_metrics = _eval_retriever(staging_dir)
            deployed = after_metrics["val_top1"] >= before_metrics["val_top1"]
            if deployed:
                if self.model_dir.exists():
                    backup_dir = self.learning_dir / "prev_sbert_bak"
                    if backup_dir.exists():
                        shutil.rmtree(backup_dir)
                    shutil.copytree(self.model_dir, backup_dir)
                    shutil.rmtree(self.model_dir)
                shutil.copytree(staging_dir, self.model_dir)
                if callable(on_model_ready):
                    on_model_ready(self.model_dir)

            report = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "device": device,
                "epochs": epochs,
                "elapsed_seconds": round(time.time() - t0, 2),
                "dataset": data_stats,
                "before_val": before_metrics,
                "after_val": after_metrics,
                "deployed": deployed,
                "preprocessing_fingerprint": preprocess_fingerprint(),
            }
            self.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            with self._lock:
                state = self._read_state()
                trained_set = set(state.get("trained_intents", []))
                trained_set.update(intent_names)
                state["trained_intents"] = sorted(trained_set)
                state["pending_intents"] = []
                state["last_trained_at"] = report["timestamp"]
                state["last_report"] = report
                self._write_state(state)
            return report
        except Exception as exc:
            self._last_error = str(exc)
            raise
        finally:
            with self._lock:
                self._running = False
