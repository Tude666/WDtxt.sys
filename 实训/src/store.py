"""文档注册表：记录已入库文档的元信息，持久化到 JSON 文件。

每条记录形如 {"id", "name", "chunk_count", "added_at"}。
模块保持纯逻辑，不依赖 Streamlit，便于单测。
"""
import json

import config

_REGISTRY_PATH = config.BASE_DIR / "documents.json"


def _load() -> list[dict]:
    """读取注册表；文件不存在或损坏时返回空列表。"""
    if not _REGISTRY_PATH.exists():
        return []
    try:
        data = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save(docs: list[dict]) -> None:
    _REGISTRY_PATH.write_text(
        json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def list_documents() -> list[dict]:
    """返回按入库时间倒序的文档列表（最新在前）。"""
    docs = _load()
    docs.sort(key=lambda d: d.get("added_at", ""), reverse=True)
    return docs


def get_document(doc_id: str) -> dict | None:
    for d in _load():
        if d.get("id") == doc_id:
            return d
    return None


def register_document(meta: dict) -> None:
    docs = _load()
    docs.append(meta)
    _save(docs)


def remove_document(doc_id: str) -> None:
    _save([d for d in _load() if d.get("id") != doc_id])
