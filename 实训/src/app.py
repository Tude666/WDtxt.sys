"""Streamlit 入口：多文档管理 + 多轮问答，流式展示回答与引用来源。"""
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

import config
import rag
import store
from document_loader import load_document

st.set_page_config(page_title="文档问答系统", page_icon="📄", layout="wide")


@st.cache_resource
def get_embeddings_cached():
    """缓存嵌入模型对象，避免重复加载。"""
    return rag.get_embeddings()


@st.cache_resource
def get_vectorstore_cached():
    """缓存统一文档集合（所有文档共享同一集合，靠 doc_id 隔离）。"""
    return rag.get_vectorstore(embeddings=get_embeddings_cached())


def ingest_upload(uploaded_file) -> tuple[bool, str]:
    """解析上传文件并作为新文档入库，返回 (是否成功, 提示信息)。"""
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        docs = load_document(tmp_path, source_name=uploaded_file.name)
        if not docs or all(not d.page_content.strip() for d in docs):
            return False, "文档内容为空或无法解析，请更换文件。"
        doc_id = uuid.uuid4().hex[:12]
        n = rag.add_documents(get_vectorstore_cached(), docs, doc_id)
        store.register_document({
            "id": doc_id,
            "name": uploaded_file.name,
            "chunk_count": n,
            "added_at": datetime.now().isoformat(timespec="seconds"),
        })
        return True, f"入库成功：{uploaded_file.name}（{n} 个片段）"
    finally:
        os.unlink(tmp_path)


st.title("📄 基于 LangChain 的文档问答系统")
st.caption("上传多份 PDF / Word 文档，入库后即可针对文档内容提问。")

# ---- 侧边栏：文档管理 ----
with st.sidebar:
    st.header("📚 文档库")
    # 用递增 key 实现"入库成功后清空上传框"：key 变了组件视为全新，旧文件自动丢弃
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0
    uploaded_file = st.file_uploader(
        "上传文档",
        type=["pdf", "docx"],
        key=f"uploader_{st.session_state['uploader_key']}",
    )
    if st.button("解析并入库", type="primary", disabled=uploaded_file is None):
        ok, msg = ingest_upload(uploaded_file)
        if ok:
            st.toast(msg)
            st.session_state["uploader_key"] += 1  # 换 key 以清空上传框
            st.rerun()
        else:
            st.error(msg)

    st.divider()
    docs = store.list_documents()
    if not docs:
        st.caption("暂无文档，请先上传。")
    else:
        for d in docs:
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**{d['name']}**")
            c1.caption(f"{d['chunk_count']} 片段 · {d['added_at'][:16].replace('T', ' ')}")
            if c2.button("🗑️", key=f"del_{d['id']}", help="删除该文档"):
                rag.delete_document(get_vectorstore_cached(), d["id"])
                store.remove_document(d["id"])
                st.toast(f"已删除：{d['name']}")
                st.rerun()

    st.divider()
    # 检索范围：全部文档或指定某篇
    doc_ids = [d["id"] for d in docs]
    scope = st.selectbox(
        "检索范围",
        options=["__all__"] + doc_ids,
        format_func=lambda x: "全部文档" if x == "__all__" else store.get_document(x)["name"],
    )
    scope_doc_id = None if scope == "__all__" else scope

# ---- 问答区（多轮 + 流式）----
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("输入你的问题"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if not store.list_documents():
            reply = "请先在左侧上传并入库文档。"
            st.write(reply)
        else:
            try:
                history = st.session_state.messages[:-1]
                with st.spinner("正在检索……"):
                    stream, docs = rag.stream_answer(
                        get_vectorstore_cached(),
                        question,
                        history=history,
                        doc_id=scope_doc_id,
                    )
                reply = st.write_stream(stream)
                with st.expander("🔎 引用来源片段"):
                    for i, doc in enumerate(docs, 1):
                        source = doc.metadata.get("source", "未知来源")
                        page = doc.metadata.get("page")
                        label = source if page is None else f"{source} · 第 {page} 页"
                        st.markdown(f"**[{i}]** {label}")
                        st.markdown(doc.page_content[:300])
                        st.markdown("---")
            except Exception as e:
                reply = f"回答失败：{e}"
                st.error(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
