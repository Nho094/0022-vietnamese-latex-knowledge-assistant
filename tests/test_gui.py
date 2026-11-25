"""Regression checks cho local Knowledge Manager và điểm nối API."""

from __future__ import annotations

import shutil
import subprocess
import re
from collections import Counter

import pytest

from api import server
from core.config import GUI_MOUNT_PATH, GUI_STATIC_DIR


def test_gui_is_mounted_and_keeps_legacy_api_routes() -> None:
    paths = {getattr(route, "path", "") for route in server.app.routes}

    assert "/" in paths
    assert GUI_MOUNT_PATH in paths
    assert "/ask" in paths
    assert "/healthz" in paths
    assert "/knowledge/definitions" in paths
    assert "/knowledge/corpus" in paths


def test_gui_assets_cover_dashboard_and_definition_wizard() -> None:
    html = (GUI_STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (GUI_STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'id="definition-rows"' in html
    assert 'id="corpus-list"' in html
    assert 'id="wizard-content"' in html
    assert 'api("/definitions")' in script
    assert 'api("/corpus")' in script
    assert 'api("/validate"' in script
    assert "function collectSchemaField" in script
    assert "function collectPolicies" in script
    assert 'index > state.step ? "disabled"' in script
    assert 'id="import-dialog"' in html
    assert 'id="execute-query-button"' in html
    assert "function previewCurrentDefinition" in script
    assert 'api("/import/validate"' in script
    assert 'api("/rollback"' in script
    assert 'api("/export"' in script
    assert "state.previewedDefinition !== JSON.stringify(definition)" in script


def test_gui_html_has_unique_ids_and_local_stylesheet() -> None:
    html = (GUI_STATIC_DIR / "index.html").read_text(encoding="utf-8")
    ids = re.findall(r'\bid="([^"]+)"', html)
    duplicate_ids = [item for item, count in Counter(ids).items() if count > 1]

    assert not duplicate_ids
    assert 'href="/ui/management.css"' in html
    assert (GUI_STATIC_DIR / "management.css").is_file()


def test_gui_javascript_parses_when_node_is_available() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js chưa cài; bỏ qua kiểm tra cú pháp client JavaScript.")

    subprocess.run(
        [node, "--check", str(GUI_STATIC_DIR / "app.js")],
        check=True,
        capture_output=True,
        text=True,
    )


def test_corpus_catalog_is_read_only_and_separate_from_intents() -> None:
    catalog = server.list_knowledge_corpora()

    assert catalog["corpora"]
    assert all("path" in corpus and "role" in corpus for corpus in catalog["corpora"])
