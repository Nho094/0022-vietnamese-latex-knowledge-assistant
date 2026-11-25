"""KA-21: pipeline NLU giữ nguyên, processor v2 chạy nhánh opt-in."""

from __future__ import annotations

from types import SimpleNamespace

from core.config import (
    ENGINE_VERSION_GENERIC,
    ENGINE_VERSION_LEGACY,
    MODE_PHOBERT_ONLY,
    MODE_RETRIEVAL_ONLY,
)
from core.pipeline import Pipeline
from core.retrieval import MockSentenceTransformer, Retriever


class FixedClassifier:
    """Classifier fixture để kiểm tra routing mà không nạp checkpoint."""

    def __init__(self, intent: str, confidence: float, category: str = "tex_doc") -> None:
        self.intent = intent
        self.confidence = confidence
        self.category = category

    def predict(self, _text: str) -> SimpleNamespace:
        return SimpleNamespace(
            intent=self.intent,
            confidence=self.confidence,
            category=self.category,
            all_probabilities={self.intent: self.confidence},
        )


def test_opt_in_v2_pipeline_runs_existing_classifier_then_generic_stages() -> None:
    pipeline = Pipeline(
        mode=MODE_PHOBERT_ONLY,
        engine_version=ENGINE_VERSION_GENERIC,
        classifier=FixedClassifier("LATEX_NAP_PACKAGE", 0.97),
        tau=0.8,
    )

    result = pipeline.predict("Nạp package amsmath")
    payload = result.to_dict()

    assert result.status == "ok"
    assert result.intent == "LATEX_NAP_PACKAGE"
    assert result.latex_code == r"\usepackage{ amsmath }"
    assert set(payload["stages"]) == {
        "intent_detection",
        "extraction",
        "field_policy",
        "normalization",
        "validation",
        "transform",
        "render",
    }
    assert payload["stages"]["intent_detection"]["source"] == "classifier_or_retrieval"
    assert payload["outputs"]["latex"] == result.latex_code


def test_v2_need_slots_can_continue_with_manual_values_without_rerouting() -> None:
    pipeline = Pipeline(
        mode=MODE_PHOBERT_ONLY,
        engine_version=ENGINE_VERSION_GENERIC,
        classifier=FixedClassifier("LATEX_NAP_PACKAGE", 0.97),
        tau=0.8,
    )

    incomplete = pipeline.predict("Nạp package")
    resumed = pipeline.choose(
        "LATEX_NAP_PACKAGE",
        query=incomplete.query,
        slots={"package_name": "amsmath"},
    )

    assert incomplete.status == "need_slots"
    assert incomplete.missing_slots == ["$.package_name"]
    assert resumed.status == "ok"
    assert resumed.latex_code == r"\usepackage{ amsmath }"


def test_legacy_pipeline_output_shape_and_route_remain_unchanged() -> None:
    pipeline = Pipeline(
        mode=MODE_PHOBERT_ONLY,
        engine_version=ENGINE_VERSION_LEGACY,
        classifier=FixedClassifier("LATEX_NAP_PACKAGE", 0.97),
        tau=0.8,
    )

    result = pipeline.predict("Nạp package amsmath")
    payload = result.to_dict()

    assert result.status == "ok"
    assert result.latex_code == r"\usepackage{amsmath}"
    assert "stages" not in payload
    assert "outputs" not in payload


def test_v2_pipeline_supports_render_target_declared_only_in_data() -> None:
    pipeline = Pipeline(
        mode=MODE_PHOBERT_ONLY,
        target="obsidian",
        engine_version=ENGINE_VERSION_GENERIC,
        classifier=FixedClassifier("OBSIDIAN_OBSIDIAN_BASIC_TIEU_E_1", 0.97, "obsidian"),
        tau=0.8,
    )

    result = pipeline.predict("Nạp package amsmath")

    assert result.status == "ok"
    assert "obsidian" in result.outputs


def test_new_registry_definition_is_retrieved_and_processed_without_classifier_retraining(tmp_path) -> None:
    definition = {
        "intent": "NEVER_TRAINED_INTENT",
        "definition_version": "2.0",
        "aliases": ["orbital matrix format"],
        "describe": "Render an orbital matrix format",
        "category": "new-category",
        "domain": "new-domain",
        "input_schema": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        "field_policies": {
            "value": {
                "value_type": "string",
                "extraction": {"method": "manual"},
                "required": True,
                "on_missing": {"action": "ask_user"},
            }
        },
        "transform": {"engine": "jsonata", "expression": "$"},
        "render": {"engine": "jinja2", "templates": {"latex": r"\text{ {{ value }} }"}},
        "slots": {},
        "templates": {"latex": "", "python": "", "cpp": ""},
    }
    retriever = Retriever(
        model=MockSentenceTransformer(embedding_dim=16),
        intents=[],
        cache_dir=tmp_path / "cache",
        use_cache=False,
    )
    retriever.add_intent(definition)
    pipeline = Pipeline(
        mode=MODE_RETRIEVAL_ONLY,
        retriever=retriever,
        auto_accept_retrieval=True,
        theta=-1,
    )
    pipeline.sync_definition_snapshot(
        {definition["intent"]: definition}, {definition["intent"]}
    )

    detected = pipeline.predict("orbital matrix format")
    completed = pipeline.choose(
        "NEVER_TRAINED_INTENT", query="orbital matrix format", slots={"value": "new"}
    )

    assert detected.intent == "NEVER_TRAINED_INTENT"
    assert detected.status == "need_slots"
    assert completed.status == "ok"
    assert completed.latex_code == r"\text{ new }"
