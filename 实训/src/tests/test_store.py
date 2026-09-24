"""store.py 单元测试：文档注册表的增删查。"""
import store


def test_register_list_remove(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_REGISTRY_PATH", tmp_path / "documents.json")

    store.register_document(
        {"id": "1", "name": "a.pdf", "chunk_count": 3, "added_at": "2026-01-01T00:00:00"}
    )
    store.register_document(
        {"id": "2", "name": "b.pdf", "chunk_count": 5, "added_at": "2026-01-02T00:00:00"}
    )

    # 最新在前
    assert [d["id"] for d in store.list_documents()] == ["2", "1"]
    assert store.get_document("1")["name"] == "a.pdf"
    assert store.get_document("missing") is None

    store.remove_document("1")
    assert [d["id"] for d in store.list_documents()] == ["2"]
    assert store.get_document("1") is None


def test_load_returns_empty_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_REGISTRY_PATH", tmp_path / "nope.json")
    assert store.list_documents() == []
