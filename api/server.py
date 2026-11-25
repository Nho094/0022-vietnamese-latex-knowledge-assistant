"""API suy luận tương thích và management service cho Contract v2 definitions."""
from __future__ import annotations

import hashlib
import hmac
import importlib.util
import sys
import json
import os
import re
from pathlib import Path
from typing import Any, Literal
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cli.ka import create_pipeline
from core.config import (
    DEFAULT_INTENTS_PATH,
    DEFAULT_CHECKPOINT_DIR,
    GUI_MOUNT_PATH,
    GUI_STATIC_DIR,
    KNOWLEDGE_CATALOG_PATH,
    INDEX_CACHE_DIR,
    MODEL_ROOT,
    PHOBERT_DIR,
    SBERT_DIR,
    SBERT_MODEL_NAME,
    DEFINITION_BACKUP_SUFFIX,
    TARGET_LATEX,
    classifier_artifact_status,
    resolve_phobert_tokenizer_dir,
    retrieval_model_status,
)
from core.retrieval import inspect_index_cache
from core.definitions import DefinitionLoader, DefinitionRegistry, DefinitionRegistryError
from core.engine import GenericKnowledgeEngine
from core.validator import slugify_intent_name

app = FastAPI(
    title="Vietnamese LaTeX Knowledge Assistant",
    openapi_url=None,
    docs_url=None,
    redoc_url=None,
)


@app.get("/", include_in_schema=False)
def gui_home() -> RedirectResponse:
    """Mở local knowledge manager ở path đã cấu hình."""
    return RedirectResponse(url=f"{GUI_MOUNT_PATH}/")


if GUI_STATIC_DIR.is_dir():
    app.mount(GUI_MOUNT_PATH, StaticFiles(directory=GUI_STATIC_DIR, html=True), name="knowledge-gui")

RUNTIME_INTENTS_PATH = Path(os.environ.get("KA_INTENTS_PATH", str(DEFAULT_INTENTS_PATH)))
pipeline = create_pipeline(intents_path=RUNTIME_INTENTS_PATH)
definition_registry: DefinitionRegistry | None = None


class AskRequest(BaseModel):
    text: str
    target: str = TARGET_LATEX


class DefinitionRequest(BaseModel):
    definition: dict[str, Any]


class DefinitionImportRequest(BaseModel):
    definitions: list[dict[str, Any]]


class DefinitionPublishRequest(BaseModel):
    definition: dict[str, Any]
    overwrite: bool = False


class DefinitionPreviewRequest(BaseModel):
    definition: dict[str, Any]
    values: dict[str, Any] = Field(default_factory=dict)
    target: str = TARGET_LATEX


class RollbackRequest(BaseModel):
    version: str | None = None


class BuilderField(BaseModel):
    key: str
    label: str
    value_type: Literal["string", "number", "integer", "boolean"] = "string"
    required: bool = False
    has_default: bool = False
    default: Any = None


class BuilderSegment(BaseModel):
    kind: Literal["text", "field"]
    value: str


class BuilderCompileRequest(BaseModel):
    name: str
    description: str
    category: str
    examples: list[str] = Field(default_factory=list)
    fields: list[BuilderField] = Field(default_factory=list)
    segments: list[BuilderSegment] = Field(default_factory=list)
    target: Literal["latex", "mathjax", "obsidian"] = "latex"
    intent: str | None = None
    base_definition: dict[str, Any] | None = None


@app.middleware("http")
async def require_extension_session(request, call_next):
    """Bảo vệ service do Extension khởi chạy bằng token cục bộ theo phiên."""
    expected = os.environ.get("KA_SESSION_TOKEN", "")
    if expected and request.url.path != "/healthz":
        supplied = request.headers.get("X-KA-Session", "")
        if not hmac.compare_digest(supplied, expected):
            return JSONResponse(status_code=401, content={"detail": "Phiên Extension không hợp lệ."})
    return await call_next(request)


def _get_definition_registry() -> DefinitionRegistry:
    global definition_registry
    if definition_registry is None:
        registry_path = Path(os.environ.get("KA_INTENTS_PATH", str(DEFAULT_INTENTS_PATH)))
        definition_registry = DefinitionRegistry(registry_path)
    return definition_registry


def _sync_pipeline_registry(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> None:
    """Expose changed definitions to the pipeline without updating PhoBERT."""
    if not hasattr(pipeline, "sync_definition_snapshot"):
        return
    previous_dynamic = set(getattr(pipeline, "_dynamic_generic_intents", set()))
    changed = {
        name
        for name, definition in after.items()
        if name not in before or before[name] != definition
    }
    dynamic_intents = (previous_dynamic & after.keys()) | changed
    pipeline.sync_definition_snapshot(after, dynamic_intents)


def _ensure_registry_retriever(registry: DefinitionRegistry) -> None:
    if registry.retriever is not None:
        return
    if hasattr(pipeline, "_ensure_components_loaded"):
        pipeline._ensure_components_loaded()
    if getattr(pipeline, "retriever", None) is None and hasattr(pipeline, "_init_default_retriever"):
        pipeline._init_default_retriever()
    retriever = getattr(pipeline, "retriever", None)
    if retriever is None:
        raise HTTPException(status_code=503, detail="Tìm kiếm cục bộ chưa sẵn sàng; dữ liệu chưa được thay đổi.")
    try:
        registry.attach_retriever(retriever)
    except DefinitionRegistryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _validation_payload(definitions: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        loaded = DefinitionLoader().load(definitions)
    except DefinitionRegistryError as exc:
        return {"valid": False, "errors": [{"path": "$", "message": str(exc)}]}
    except ValueError as exc:
        return {"valid": False, "errors": [{"path": "$", "message": str(exc)}]}
    legacy = [item.intent for item in loaded if item.source_contract != "v2"]
    if legacy:
        return {
            "valid": False,
            "errors": [
                {"path": f"{intent}", "message": "Import/publish chỉ nhận Contract v2."}
                for intent in legacy
            ],
        }
    return {
        "valid": True,
        "count": len(loaded),
        "intents": [item.intent for item in loaded],
        "errors": [],
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/system/status")
def system_status() -> dict[str, Any]:
    """Trả trạng thái môi trường cho Extension mà không nạp hoặc tải model."""
    classifier_checkpoint, classifier_message = classifier_artifact_status(DEFAULT_CHECKPOINT_DIR)
    classifier_dependencies = ("torch", "transformers", "pyvi")
    missing_classifier_dependencies = [
        dependency for dependency in classifier_dependencies
        if importlib.util.find_spec(dependency) is None
    ]
    if (DEFAULT_CHECKPOINT_DIR / "model.safetensors").is_file() and not (
        DEFAULT_CHECKPOINT_DIR / "pytorch_model.bin"
    ).is_file() and importlib.util.find_spec("safetensors") is None:
        missing_classifier_dependencies.append("safetensors")
    classifier_dependencies_available = not missing_classifier_dependencies
    if classifier_checkpoint and missing_classifier_dependencies:
        classifier_checkpoint = False
    if missing_classifier_dependencies:
        classifier_message = "Thiếu thư viện PhoBERT: " + ", ".join(missing_classifier_dependencies)

    retrieval_model, retrieval_message, retrieval_path = retrieval_model_status()
    missing_retrieval_dependencies = (
        ["sentence-transformers"]
        if importlib.util.find_spec("sentence_transformers") is None
        else []
    )
    retrieval_dependencies_available = not missing_retrieval_dependencies
    if missing_retrieval_dependencies:
        retrieval_model = False
        retrieval_message = "Thiếu thư viện Python sentence-transformers."
    retrieval_index = False
    retrieval_index_message = "Model retrieval chưa sẵn sàng; index chưa được kiểm tra."
    cache_path: Path | None = None
    if retrieval_model:
        try:
            intents = json.loads(RUNTIME_INTENTS_PATH.read_text(encoding="utf-8"))
            retrieval_index, retrieval_index_message, cache_path = inspect_index_cache(
                INDEX_CACHE_DIR, intents, model_name=SBERT_MODEL_NAME
            )
        except (OSError, ValueError, TypeError) as error:
            retrieval_index_message = f"Chưa kiểm tra được index cache: {error}"

    both_ready = classifier_checkpoint and retrieval_model
    model_details = [classifier_message, retrieval_message, retrieval_index_message]
    if both_ready:
        message = "PhoBERT và retrieval model sẵn sàng. " + retrieval_index_message
    elif classifier_checkpoint:
        message = f"PhoBERT sẵn sàng; đang dùng classifier-only. Retrieval: {retrieval_message}"
    elif retrieval_model:
        message = f"Retrieval model sẵn sàng; đang dùng retrieval-only. PhoBERT: {classifier_message}"
    else:
        message = (
            "Backend và Knowledge Manager hoạt động; chưa có model NLP khả dụng. "
            + " ".join(model_details[:2])
        )
    return {
        "status": "ready" if both_ready else "degraded",
        "python": sys.version.split()[0],
        "classifier_available": classifier_checkpoint,
        "classifier_dependencies_available": classifier_dependencies_available,
        "classifier_missing_dependencies": missing_classifier_dependencies,
        "classifier_message": classifier_message,
        "classifier_checkpoint_path": str(DEFAULT_CHECKPOINT_DIR),
        "phobert_base_path": str(resolve_phobert_tokenizer_dir() or PHOBERT_DIR),
        "retrieval_model_available": retrieval_model,
        "retrieval_dependencies_available": retrieval_dependencies_available,
        "retrieval_missing_dependencies": missing_retrieval_dependencies,
        "retrieval_model_message": retrieval_message,
        "retrieval_model_path": str(retrieval_path or SBERT_DIR),
        "retrieval_index_available": retrieval_index,
        "retrieval_cache_available": retrieval_index,
        "retrieval_index_message": retrieval_index_message,
        "retrieval_index_path": str(cache_path) if cache_path is not None else str(INDEX_CACHE_DIR),
        "model_root": str(MODEL_ROOT),
        "builder_model_available": False,
        "message": message,
    }


@app.post("/ask")
def ask_endpoint(request: AskRequest) -> dict[str, object]:
    selected_pipeline = pipeline if request.target == pipeline.target else create_pipeline(
        mode=pipeline.mode, target=request.target
    )
    result = selected_pipeline.predict(request.text)
    return result.to_dict()


class ChooseRequest(BaseModel):
    text: str
    candidate: str
    values: dict[str, Any] = Field(default_factory=dict)
    target: str = TARGET_LATEX


@app.post("/ask/choose")
def choose_endpoint(request: ChooseRequest) -> dict[str, object]:
    """Tiếp tục một yêu cầu sau khi Extension nhận lựa chọn và giá trị còn thiếu."""
    selected_pipeline = pipeline if request.target == pipeline.target else create_pipeline(
        mode=pipeline.mode, target=request.target
    )
    result = selected_pipeline.choose(
        request.candidate,
        query=request.text,
        slots=request.values,
    )
    return result.to_dict()


@app.get("/knowledge/definitions")
def list_definitions(
    search: str = "",
    category: str | None = None,
    domain: str | None = None,
) -> dict[str, Any]:
    """List definitions with optional text/category/domain filters."""
    items = list(_get_definition_registry().snapshot().values())
    search_key = search.strip().casefold()
    filtered = [
        item
        for item in items
        if (not search_key or search_key in " ".join(
            [str(item.get("intent", "")), str(item.get("describe", "")), *map(str, item.get("aliases", []))]
        ).casefold())
        and (category is None or item.get("category") == category)
        and (domain is None or item.get("domain") == domain)
    ]
    return {
        "count": len(filtered),
        "definitions": filtered,
        "categories": sorted({str(item.get("category", "")) for item in items if item.get("category")}),
        "domains": sorted({str(item.get("domain", "")) for item in items if item.get("domain")}),
    }


@app.get("/knowledge/corpus")
def list_knowledge_corpora() -> dict[str, Any]:
    """Read the separate corpus catalog without mixing it with intent data."""
    try:
        catalog = json.loads(KNOWLEDGE_CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Không đọc được knowledge catalog: {exc}") from exc
    return catalog


@app.get("/knowledge/definitions/{intent}")
def get_definition(intent: str) -> dict[str, Any]:
    definition = _get_definition_registry().get(intent)
    if definition is None:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy intent '{intent}'.")
    return definition


@app.post("/knowledge/validate")
def validate_definition(request: DefinitionRequest) -> dict[str, Any]:
    return _validation_payload([request.definition])


@app.post("/knowledge/import/validate")
def validate_import(request: DefinitionImportRequest) -> dict[str, Any]:
    """Validate import batch only; this route never writes or overwrites files."""
    result = _validation_payload(request.definitions)
    if result["valid"]:
        existing = _get_definition_registry().snapshot()
        conflicts = sorted(set(result["intents"]).intersection(existing))
        result["conflicts"] = conflicts
        result["requires_overwrite_confirmation"] = bool(conflicts)
    return result


class DefinitionImportPublishRequest(DefinitionImportRequest):
    overwrite: bool = False


@app.post("/knowledge/import")
def import_definitions(request: DefinitionImportPublishRequest) -> dict[str, Any]:
    """Import một batch đã validate, yêu cầu xác nhận rõ nếu trùng kiến thức."""
    if not request.definitions:
        raise HTTPException(status_code=422, detail="Tệp nhập không có kiến thức nào.")
    validation = _validation_payload(request.definitions)
    if not validation["valid"]:
        raise HTTPException(status_code=422, detail=validation)
    registry = _get_definition_registry()
    existing = registry.snapshot()
    conflicts = sorted({item["intent"] for item in request.definitions}.intersection(existing))
    if conflicts and not request.overwrite:
        raise HTTPException(
            status_code=409,
            detail={"conflicts": conflicts, "requires_overwrite_confirmation": True},
        )
    _ensure_registry_retriever(registry)
    try:
        before = registry.snapshot()
        version, intents = registry.publish_many(request.definitions, overwrite=request.overwrite)
        _sync_pipeline_registry(before, registry.snapshot())
        return {"imported": True, "count": len(intents), "intents": intents, "version": version}
    except DefinitionRegistryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/knowledge/export")
def export_definitions(intent: str | None = None) -> dict[str, Any]:
    registry = _get_definition_registry()
    if intent is not None:
        definition = registry.get(intent)
        if definition is None:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy intent '{intent}'.")
        return {"definitions": [definition], "count": 1}
    definitions = list(registry.snapshot().values())
    return {"definitions": definitions, "count": len(definitions)}


@app.post("/knowledge/preview")
def preview_definition(request: DefinitionPreviewRequest) -> dict[str, Any]:
    validation = _validation_payload([request.definition])
    if not validation["valid"]:
        raise HTTPException(status_code=422, detail=validation)
    result = GenericKnowledgeEngine().process(
        "",
        request.definition,
        target=request.target,
        manual_values=request.values,
    )
    return {
        "status": result.status,
        "values": result.values,
        "missing": result.missing,
        "errors": [error.__dict__ for error in result.errors],
        "outputs": result.outputs,
        "stages": result.stages,
    }


def _escape_builder_literal(value: str) -> str:
    """Đưa chữ tĩnh vào literal Jinja để dấu ngoặc LaTeX không thành cú pháp."""
    return "{{ " + json.dumps(value, ensure_ascii=False) + " }}"


def _compile_builder_definition(request: BuilderCompileRequest) -> dict[str, Any]:
    name = request.name.strip()
    description = request.description.strip()
    category = request.category.strip()
    if not name or not description or not category:
        raise HTTPException(status_code=422, detail="Cần nhập tên, mô tả và nhóm kiến thức.")
    intent = (request.intent or slugify_intent_name(name)).strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", intent):
        raise HTTPException(status_code=422, detail="Tên kiến thức chưa hợp lệ.")
    keys: set[str] = set()
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    policies: dict[str, dict[str, Any]] = {}
    for field in request.fields:
        key = field.key.strip()
        label = field.label.strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key) or key in keys or not label:
            raise HTTPException(status_code=422, detail=f"Trường '{label or key}' cần tên riêng biệt.")
        keys.add(key)
        property_schema: dict[str, Any] = {"type": field.value_type, "title": label}
        if field.has_default:
            property_schema["default"] = field.default
        properties[key] = property_schema
        if field.required:
            required.append(key)
        missing_action = "use_default" if field.has_default else ("ask_user" if field.required else "omit")
        policies[key] = {
            "description": label,
            "value_type": field.value_type,
            "extraction": {"method": "manual"},
            "required": field.required,
            "on_missing": {
                "action": missing_action,
                **({"value": field.default} if field.has_default else {}),
            },
        }

    template_parts: list[str] = []
    for segment in request.segments:
        if segment.kind == "text":
            template_parts.append(_escape_builder_literal(segment.value))
            continue
        if segment.value not in keys:
            raise HTTPException(status_code=422, detail="Mẫu kết quả tham chiếu một trường không tồn tại.")
        value_type = next(field.value_type for field in request.fields if field.key == segment.value)
        filter_suffix = " | latex_escape" if request.target == "latex" and value_type == "string" else ""
        template_parts.append("{{ " + segment.value + filter_suffix + " }}")
    if not request.segments:
        raise HTTPException(status_code=422, detail="Hãy thêm nội dung cho mẫu kết quả.")

    examples = list(dict.fromkeys(value.strip() for value in request.examples if value.strip()))
    template = "".join(template_parts)
    base_definition = request.base_definition or {}
    old_render = base_definition.get("render", {})
    old_templates = old_render.get("templates", {}) if isinstance(old_render, dict) else {}
    render = {
        "engine": "jinja2",
        "context": "normalized_input",
        "templates": {**old_templates, request.target: template},
    }
    definition = {
        **base_definition,
        "intent": intent,
        "display_name": name,
        "definition_version": "2.0",
        "aliases": examples or [name],
        "describe": description,
        "category": category,
        "domain": "user-created",
        "input_schema": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        "field_policies": policies,
        "transform": base_definition.get(
            "transform", {"engine": "jsonata", "expression": "$", "mode": "identity"}
        ),
        "render": render,
    }
    return definition


@app.post("/knowledge/builder/compile")
def compile_builder_definition(request: BuilderCompileRequest) -> dict[str, Any]:
    """Chuyển biểu mẫu no-code thành Contract v2; route này không ghi dữ liệu."""
    definition = _compile_builder_definition(request)
    validation = _validation_payload([definition])
    return {"valid": validation["valid"], "errors": validation["errors"], "definition": definition}


@app.get("/knowledge/builder/capabilities")
def builder_capabilities() -> dict[str, Any]:
    """Báo rõ khi không có model sinh đề xuất; builder thủ công luôn dùng được."""
    return {
        "suggestions_available": False,
        "manual_builder_available": True,
        "message": "Gợi ý bằng AI chưa được tích hợp; biểu mẫu tạo kiến thức thủ công vẫn sẵn sàng.",
    }


@app.post("/knowledge/publish")
def publish_definition(request: DefinitionPublishRequest) -> dict[str, Any]:
    validation = _validation_payload([request.definition])
    if not validation["valid"]:
        raise HTTPException(status_code=422, detail=validation)
    registry = _get_definition_registry()
    _ensure_registry_retriever(registry)
    try:
        before = registry.snapshot()
        version = registry.publish(request.definition, overwrite=request.overwrite)
        _sync_pipeline_registry(before, registry.snapshot())
    except DefinitionRegistryError as exc:
        message = str(exc)
        status = 409 if "overwrite=True" in message else 422
        raise HTTPException(status_code=status, detail=message) from exc
    return {"published": True, "intent": request.definition["intent"], "version": version}


@app.post("/knowledge/reload")
def reload_definitions() -> dict[str, Any]:
    registry = _get_definition_registry()
    _ensure_registry_retriever(registry)
    try:
        before = registry.snapshot()
        version = registry.reload()
        _sync_pipeline_registry(before, registry.snapshot())
        return {"reloaded": True, "version": version}
    except DefinitionRegistryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/knowledge/rollback")
def rollback_definitions(request: RollbackRequest) -> dict[str, Any]:
    registry = _get_definition_registry()
    _ensure_registry_retriever(registry)
    try:
        before = registry.snapshot()
        version = registry.rollback(request.version)
        _sync_pipeline_registry(before, registry.snapshot())
        return {"rolled_back": True, "version": version}
    except DefinitionRegistryError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.delete("/knowledge/definitions/{intent}")
def delete_definition(intent: str) -> dict[str, Any]:
    """Xóa qua registry để giữ backup trước khi thay bank và retrieval index."""
    if intent == "OUT_OF_SCOPE":
        raise HTTPException(status_code=409, detail="Không thể xóa kiến thức hệ thống ngoài phạm vi.")
    registry = _get_definition_registry()
    if registry.get(intent) is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy kiến thức.")
    _ensure_registry_retriever(registry)
    try:
        before = registry.snapshot()
        version = registry.remove(intent)
        _sync_pipeline_registry(before, registry.snapshot())
        return {"deleted": True, "intent": intent, "version": version}
    except DefinitionRegistryError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/knowledge/history")
def definition_history() -> dict[str, Any]:
    """Liệt kê các snapshot backup có thể phục hồi sau khi khởi động lại."""
    registry = _get_definition_registry()
    pattern = f"{registry.path.name}.*{DEFINITION_BACKUP_SUFFIX}"
    versions = []
    for path in sorted(registry.path.parent.glob(pattern), key=lambda item: item.stat().st_mtime_ns, reverse=True):
        version = path.name[len(registry.path.name) + 1 : -len(DEFINITION_BACKUP_SUFFIX)]
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            stat = path.stat()
        except OSError:
            continue
        versions.append({
            "version": version,
            "created_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
            "active": version == registry.version,
            "valid_hash": digest == version,
        })
    return {"active_version": registry.version, "versions": versions}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
