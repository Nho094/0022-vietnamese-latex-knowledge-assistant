"""Atomic in-process definition registry phối hợp durable JSON và SBERT index."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from threading import RLock
from typing import Any

from core.config import DEFAULT_INTENTS_PATH, DEFINITION_BACKUP_SUFFIX
from core.definitions.loader import DefinitionLoader


class DefinitionRegistryError(RuntimeError):
    """Publish/reload/rollback bị từ chối hoặc không thể hoàn tất an toàn."""


def _content_version(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stable_payload(definitions: Mapping[str, Mapping[str, Any]]) -> bytes:
    ordered = [copy.deepcopy(dict(definition)) for definition in definitions.values()]
    return (json.dumps(ordered, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


class DefinitionRegistry:
    """Immutable-read snapshots, validated publish, backup and rollback.

    Retriever operations are protected by its shared index lock so reads never
    observe rows and embeddings from different generations. File replacement is
    atomic; an index change is reverted if persistence fails.
    """

    def __init__(
        self,
        path: Path | str = DEFAULT_INTENTS_PATH,
        *,
        retriever: Any | None = None,
        loader: DefinitionLoader | None = None,
    ) -> None:
        self.path = Path(path)
        self.retriever = retriever
        self.loader = loader or DefinitionLoader()
        self._lock = RLock()
        self._snapshot, data = self._read_snapshot()
        self._version = _content_version(data)
        self._history: dict[str, dict[str, dict[str, Any]]] = {
            self._version: copy.deepcopy(self._snapshot)
        }
        self._history_bytes: dict[str, bytes] = {self._version: data}
        self._versions = [self._version]

    @property
    def version(self) -> str:
        """SHA-256 generation identifier for the active durable definition bank."""
        with self._lock:
            return self._version

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return an isolated snapshot so callers cannot mutate live state."""
        with self._lock:
            return copy.deepcopy(self._snapshot)

    def get(self, intent: str) -> dict[str, Any] | None:
        """Return a detached definition by ID."""
        with self._lock:
            definition = self._snapshot.get(intent)
            return copy.deepcopy(definition) if definition is not None else None

    def publish(
        self,
        definition: Mapping[str, Any],
        *,
        overwrite: bool = False,
    ) -> str:
        """Validate, back up, index and atomically publish one v2 definition."""
        if self.retriever is None:
            raise DefinitionRegistryError("Publish cần Retriever đang hoạt động để cập nhật SBERT index.")
        try:
            loaded = self.loader.load([dict(definition)])[0]
        except (ValueError, OSError) as exc:
            raise DefinitionRegistryError(f"Definition không hợp lệ: {exc}") from exc
        if loaded.source_contract != "v2":
            raise DefinitionRegistryError("Chỉ publish definition Contract v2; legacy cần migration tường minh.")
        intent = loaded.intent
        candidate = loaded.definition

        with self._lock, self._retriever_lock():
            previous = copy.deepcopy(self._snapshot)
            exists = intent in previous
            if exists and not overwrite:
                raise DefinitionRegistryError(f"Intent '{intent}' đã tồn tại; cần overwrite=True.")
            updated = copy.deepcopy(previous)
            updated[intent] = copy.deepcopy(candidate)
            self._validate_bank(updated)
            old_index = copy.deepcopy(previous)
            try:
                self._apply_index_delta(previous, updated)
                return self._commit(updated)
            except Exception as exc:
                self._restore_index(old_index)
                if isinstance(exc, DefinitionRegistryError):
                    raise
                raise DefinitionRegistryError(f"Publish thất bại; snapshot cũ được giữ: {exc}") from exc

    def remove(self, intent: str) -> str:
        """Xóa một definition khỏi bank sau khi lưu backup và đồng bộ SBERT."""
        if self.retriever is None:
            raise DefinitionRegistryError("Xóa kiến thức cần Retriever để đồng bộ SBERT index.")
        with self._lock, self._retriever_lock():
            previous = copy.deepcopy(self._snapshot)
            if intent not in previous:
                raise DefinitionRegistryError(f"Không tìm thấy intent '{intent}'.")
            updated = copy.deepcopy(previous)
            del updated[intent]
            old_index = copy.deepcopy(previous)
            try:
                self._apply_index_delta(previous, updated)
                return self._commit(updated)
            except Exception as exc:
                self._restore_index(old_index)
                if isinstance(exc, DefinitionRegistryError):
                    raise
                raise DefinitionRegistryError(f"Xóa thất bại; snapshot cũ được giữ: {exc}") from exc

    def publish_many(
        self,
        definitions: list[Mapping[str, Any]],
        *,
        overwrite: bool = False,
    ) -> tuple[str, list[str]]:
        """Validate rồi publish cả batch trong một lần commit và một backup."""
        if self.retriever is None:
            raise DefinitionRegistryError("Import cần Retriever đang hoạt động để cập nhật SBERT index.")
        try:
            loaded = self.loader.load([dict(item) for item in definitions])
        except (ValueError, OSError) as exc:
            raise DefinitionRegistryError(f"Batch definition không hợp lệ: {exc}") from exc
        if any(item.source_contract != "v2" for item in loaded):
            raise DefinitionRegistryError("Chỉ import definition Contract v2; legacy cần migration tường minh.")
        intents = [item.intent for item in loaded]
        with self._lock, self._retriever_lock():
            previous = copy.deepcopy(self._snapshot)
            conflicts = sorted(set(intents).intersection(previous))
            if conflicts and not overwrite:
                raise DefinitionRegistryError(
                    f"Cần xác nhận ghi đè các kiến thức đã tồn tại: {', '.join(conflicts)}."
                )
            updated = copy.deepcopy(previous)
            for item in loaded:
                updated[item.intent] = copy.deepcopy(item.definition)
            self._validate_bank(updated)
            try:
                self._apply_index_delta(previous, updated)
                return self._commit(updated), intents
            except Exception as exc:
                self._restore_index(previous)
                if isinstance(exc, DefinitionRegistryError):
                    raise
                raise DefinitionRegistryError(f"Import thất bại; bank cũ được giữ: {exc}") from exc

    def reload(self) -> str:
        """Validate disk generation, reconcile SBERT and swap one snapshot."""
        with self._lock:
            updated, data = self._read_snapshot()
            version = _content_version(data)
            if version == self._version:
                return self._version
            if self.retriever is None:
                raise DefinitionRegistryError("Reload cần Retriever để đồng bộ dynamic index.")
            previous = copy.deepcopy(self._snapshot)
            with self._retriever_lock():
                try:
                    self._apply_index_delta(previous, updated)
                    self._snapshot = updated
                    self._version = version
                    self._remember(version, updated, data)
                    return version
                except Exception as exc:
                    self._restore_index(previous)
                    raise DefinitionRegistryError(f"Reload thất bại; snapshot cũ được giữ: {exc}") from exc

    def rollback(self, version: str | None = None) -> str:
        """Restore a prior validated bank version from backup/history."""
        if self.retriever is None:
            raise DefinitionRegistryError("Rollback cần Retriever để đồng bộ dynamic index.")
        with self._lock, self._retriever_lock():
            target_version = version
            if target_version is None:
                previous_versions = [item for item in self._versions if item != self._version]
                if previous_versions:
                    target_version = previous_versions[-1]
                else:
                    # Tìm backup trên đĩa để rollback vẫn hoạt động sau khi process khởi động lại.
                    backups = sorted(
                        self.path.parent.glob(
                            f"{self.path.name}.*{DEFINITION_BACKUP_SUFFIX}"
                        ),
                        key=lambda item: item.stat().st_mtime_ns,
                        reverse=True,
                    )
                    target_version = next(
                        (
                            backup.name[
                                len(self.path.name) + 1 : -len(DEFINITION_BACKUP_SUFFIX)
                            ]
                            for backup in backups
                            if backup.is_file()
                        ),
                        None,
                    )
                if target_version is None:
                    raise DefinitionRegistryError("Không có phiên bản trước để rollback.")
            if target_version == self._version:
                return self._version

            target_snapshot = self._history.get(target_version)
            target_data = self._history_bytes.get(target_version)
            if target_snapshot is None or target_data is None:
                backup = self._backup_path(target_version)
                try:
                    target_data = backup.read_bytes()
                    if _content_version(target_data) != target_version:
                        raise DefinitionRegistryError(
                            f"Backup {target_version} không khớp SHA-256 trong tên file."
                        )
                    target_snapshot = self._decode_bank(target_data)
                except (OSError, ValueError) as exc:
                    raise DefinitionRegistryError(f"Không đọc được backup {target_version}: {exc}") from exc

            current_snapshot = copy.deepcopy(self._snapshot)
            current_data = self.path.read_bytes() if self.path.exists() else _stable_payload(current_snapshot)
            self._write_backup(self._version, current_data)
            try:
                self._apply_index_delta(current_snapshot, target_snapshot)
                self._atomic_write(self.path, target_data)
                self._snapshot = copy.deepcopy(target_snapshot)
                self._version = _content_version(target_data)
                self._remember(self._version, self._snapshot, target_data)
                return self._version
            except Exception as exc:
                self._restore_index(current_snapshot)
                raise DefinitionRegistryError(f"Rollback thất bại; snapshot cũ được giữ: {exc}") from exc

    def attach_retriever(self, retriever: Any, *, synchronize: bool = True) -> None:
        """Attach SBERT and optionally rebuild it from the active definition snapshot."""
        with self._lock:
            self.retriever = retriever
            if synchronize:
                with self._retriever_lock():
                    try:
                        retriever.build_index(list(self._snapshot.values()))
                    except Exception as exc:
                        self.retriever = None
                        raise DefinitionRegistryError(f"Không đồng bộ được SBERT index: {exc}") from exc

    def _read_snapshot(self) -> tuple[dict[str, dict[str, Any]], bytes]:
        if self.path.exists():
            data = self.path.read_bytes()
            try:
                raw = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DefinitionRegistryError(f"Không đọc được definition bank: {exc}") from exc
        else:
            raw = []
            data = _stable_payload({})
        if not isinstance(raw, list):
            raise DefinitionRegistryError("Definition bank root phải là JSON array.")
        try:
            loaded = self.loader.load(raw)
        except ValueError as exc:
            raise DefinitionRegistryError(f"Definition bank không hợp lệ: {exc}") from exc
        return {item.intent: item.definition for item in loaded}, data

    def _decode_bank(self, data: bytes) -> dict[str, dict[str, Any]]:
        try:
            raw = json.loads(data.decode("utf-8"))
            loaded = self.loader.load(raw)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise DefinitionRegistryError(f"Backup không hợp lệ: {exc}") from exc
        return {item.intent: item.definition for item in loaded}

    def _validate_bank(self, definitions: Mapping[str, Mapping[str, Any]]) -> None:
        loaded = self.loader.load(list(definitions.values()))
        if len(loaded) != len(definitions):
            raise DefinitionRegistryError("Validation không trả đủ definitions trong bank.")

    def _apply_index_delta(
        self,
        previous: Mapping[str, Mapping[str, Any]],
        updated: Mapping[str, Mapping[str, Any]],
    ) -> None:
        removed = previous.keys() - updated.keys()
        for intent in removed:
            if intent in self.retriever:
                self.retriever.remove_intent(intent)
        for intent, definition in updated.items():
            if intent not in previous or previous[intent] != definition:
                self.retriever.add_intent(
                    copy.deepcopy(dict(definition)),
                    overwrite=intent in self.retriever,
                )

    def _restore_index(self, definitions: Mapping[str, Mapping[str, Any]]) -> None:
        if self.retriever is None:
            return
        try:
            self.retriever.build_index([copy.deepcopy(dict(item)) for item in definitions.values()])
        except Exception as exc:
            raise DefinitionRegistryError(f"Không thể phục hồi SBERT index: {exc}") from exc

    def _commit(self, updated: dict[str, dict[str, Any]]) -> str:
        current_data = self.path.read_bytes() if self.path.exists() else _stable_payload(self._snapshot)
        self._write_backup(self._version, current_data)
        data = _stable_payload(updated)
        self._atomic_write(self.path, data)
        version = _content_version(data)
        self._snapshot = copy.deepcopy(updated)
        self._version = version
        self._remember(version, updated, data)
        return version

    def _remember(
        self,
        version: str,
        snapshot: Mapping[str, Mapping[str, Any]],
        data: bytes,
    ) -> None:
        self._history[version] = copy.deepcopy(dict(snapshot))
        self._history_bytes[version] = data
        if version not in self._versions:
            self._versions.append(version)

    def _backup_path(self, version: str) -> Path:
        return self.path.with_name(f"{self.path.name}.{version}{DEFINITION_BACKUP_SUFFIX}")

    def _write_backup(self, version: str, data: bytes) -> None:
        backup = self._backup_path(version)
        if not backup.exists():
            self._atomic_write(backup, data)

    def _atomic_write(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def _retriever_lock(self):
        lock = getattr(self.retriever, "index_lock", None)
        return lock if lock is not None else nullcontext()
