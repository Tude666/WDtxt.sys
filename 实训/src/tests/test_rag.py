"""rag.py 单元测试：切分 / 消息转换 / 链结构 / LLM 配置。

不含网络与模型下载：嵌入与 LLM 均用桩替身，测试可离线运行。
"""
import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

import config
import rag


# ---------- split_documents ----------

def test_split_documents_splits_long_text():
    text = "第一句。" * 200  # 800 字，超过 chunk_size
    chunks = rag.split_documents([Document(page_content=text)])
    assert len(chunks) > 1
    # 每个片段不超过 chunk_size（切分器保证）
    assert all(len(c.page_content) <= config.CHUNK_SIZE for c in chunks)


def test_split_documents_preserves_metadata():
    chunks = rag.split_documents(
        [Document(page_content="短文本。", metadata={"source": "x.pdf"})]
    )
    assert chunks[0].metadata.get("source") == "x.pdf"


def test_split_documents_preserves_page():
    """切分不应丢失页码元数据（引用来源依赖它）。"""
    chunks = rag.split_documents(
        [Document(page_content="第一句。" * 100, metadata={"source": "x.pdf", "page": 3})]
    )
    assert chunks
    assert all(c.metadata.get("page") == 3 for c in chunks)


# ---------- to_messages ----------

def test_to_messages_dict_and_tuple():
    hist = [
        {"role": "user", "content": "q1"},
        ("assistant", "a1"),
        {"role": "assistant", "content": "a2"},
    ]
    msgs = rag.to_messages(hist)
    assert [m.type for m in msgs] == ["human", "ai", "ai"]
    assert msgs[0].content == "q1"
    assert msgs[2].content == "a2"


def test_to_messages_empty_and_unknown_role():
    assert rag.to_messages(None) == []
    assert rag.to_messages([]) == []
    # 未知角色应被忽略，不崩溃
    assert rag.to_messages([{"role": "system", "content": "skip"}]) == []


# ---------- _format_docs ----------

def test_format_docs_joins_with_blank_line():
    docs = [Document(page_content="a"), Document(page_content="b")]
    assert rag._format_docs(docs) == "a\n\nb"


# ---------- get_llm ----------

def test_get_llm_requires_api_key(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "")
    with pytest.raises(ValueError):
        rag.get_llm()


def test_get_llm_builds_client_with_key(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "sk-test")
    llm = rag.get_llm()
    assert llm.model_name == config.DEEPSEEK_MODEL


# ---------- build_rag_chain 结构（回归测试） ----------

def test_chain_extracts_query_string_for_retriever(monkeypatch):
    """回归：retriever 必须收到问题字符串，而非整个输入字典。

    历史上曾因 retriever 直接接收 {"input","history"} 字典，导致
    embed_query 对 dict 调 replace 报错。此测试锁定修复行为。
    """

    class FakeVS:
        def as_retriever(self, **kwargs):
            def _get(query):
                assert isinstance(query, str), f"retriever 收到非字符串: {type(query)}"
                return [Document(page_content="片段内容")]

            return RunnableLambda(_get)

    # 桩 LLM：直接返回固定字符串，跳过真实联网调用
    monkeypatch.setattr(rag, "get_llm", lambda: RunnableLambda(lambda pv: "stub"))

    chain, _ = rag.build_rag_chain(FakeVS())
    out = chain.invoke(
        {
            "input": "现在的问题",
            "history": rag.to_messages([{"role": "user", "content": "之前的问题"}]),
        }
    )
    assert out == "stub"


def test_chain_injects_history_into_prompt(monkeypatch):
    """回归：历史消息应被注入 prompt（system → human(历史) → human(当前)）。"""

    class FakeVS:
        def as_retriever(self, **kwargs):
            return RunnableLambda(lambda q: [])

    def capture(prompt_value):
        return "|".join(m.type for m in prompt_value.to_messages())

    monkeypatch.setattr(rag, "get_llm", lambda: RunnableLambda(capture))
    chain, _ = rag.build_rag_chain(FakeVS())
    out = chain.invoke(
        {
            "input": "now",
            "history": rag.to_messages([{"role": "user", "content": "prev"}]),
        }
    )
    assert out == "system|human|human"


# ---------- 多文档：add / delete / 过滤 ----------

def test_add_documents_tags_chunks_with_doc_id():
    captured = {}

    class FakeVS:
        def add_documents(self, chunks):
            captured["chunks"] = chunks

    docs = [Document(page_content="第一句。" * 200)]
    n = rag.add_documents(FakeVS(), docs, "doc123")

    assert n == len(captured["chunks"])
    assert n > 1  # 长文本应被切分
    assert all(c.metadata.get("doc_id") == "doc123" for c in captured["chunks"])


def test_delete_document_uses_where_filter():
    captured = {}

    class FakeVS:
        def delete(self, where=None):
            captured["where"] = where

    rag.delete_document(FakeVS(), "doc123")
    assert captured["where"] == {"doc_id": "doc123"}


def test_chain_applies_doc_id_filter(monkeypatch):
    captured = {}

    class FakeVS:
        def as_retriever(self, **kwargs):
            captured["search_kwargs"] = kwargs.get("search_kwargs", {})
            return RunnableLambda(lambda q: [])

    monkeypatch.setattr(rag, "get_llm", lambda: RunnableLambda(lambda pv: "stub"))

    # 指定 doc_id 时检索范围应带上过滤条件
    rag.build_rag_chain(FakeVS(), doc_id="doc1")
    assert captured["search_kwargs"].get("filter") == {"doc_id": "doc1"}

    # 不指定 doc_id 时检索全部，不带过滤
    rag.build_rag_chain(FakeVS())
    assert "filter" not in captured["search_kwargs"]
