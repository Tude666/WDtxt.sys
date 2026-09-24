"""RAG 核心：文本切分、向量化、存储、检索与生成。

模块保持纯逻辑（不依赖 Streamlit），缓存交给 app.py 用 @st.cache_resource 处理。
RAG 链使用纯 LCEL（langchain_core）实现，兼容 langchain 1.x。
"""
from operator import itemgetter

import config
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 中文友好的切分分隔符
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", " ", ""]

_SYSTEM_PROMPT = (
    "你是一名文档问答助手。仅根据以下上下文回答用户问题，"
    "不要编造上下文之外的信息；若上下文不足以回答，请明确说明"
    "“文档中未找到相关信息”。\n\n上下文：\n{context}"
)


def get_embeddings() -> HuggingFaceEmbeddings:
    """本地嵌入模型（免额外 API key，首次运行自动下载）。"""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def get_llm() -> ChatOpenAI:
    """DeepSeek 对话模型（OpenAI 兼容接口）。

    max_retries / timeout 用于应对限流与超时：底层会在 429/5xx/超时等
    可重试错误上自动退避重试，避免偶发抖动直接导致提问失败。
    """
    if not config.DEEPSEEK_API_KEY:
        raise ValueError("未配置 DEEPSEEK_API_KEY，请复制 .env.example 为 .env 并填入真实 key。")
    return ChatOpenAI(
        model=config.DEEPSEEK_MODEL,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=config.LLM_TEMPERATURE,
        timeout=config.LLM_TIMEOUT,
        max_retries=config.LLM_MAX_RETRIES,
    )


def split_documents(documents: list[Document]) -> list[Document]:
    """将文档切分为固定大小、带重叠的片段。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=_SEPARATORS,
    )
    return splitter.split_documents(documents)


def get_vectorstore(embeddings=None) -> Chroma:
    """打开（不存在则创建）统一文档集合，所有文档共享该集合。"""
    return Chroma(
        embedding_function=embeddings or get_embeddings(),
        persist_directory=config.CHROMA_DIR,
        collection_name=config.CHROMA_COLLECTION,
    )


def add_documents(vectorstore: Chroma, documents: list[Document], doc_id: str) -> int:
    """切分文档、打上 doc_id 标签后写入向量库，返回片段数。

    通过 doc_id 实现文档间隔离：检索时可用 where={"doc_id": ...} 过滤。
    """
    chunks = split_documents(documents)
    for chunk in chunks:
        chunk.metadata["doc_id"] = doc_id
    vectorstore.add_documents(chunks)
    return len(chunks)


def delete_document(vectorstore: Chroma, doc_id: str) -> None:
    """按 doc_id 删除某篇文档的全部向量片段。"""
    vectorstore.delete(where={"doc_id": doc_id})


def _format_docs(docs: list[Document]) -> str:
    """将检索到的文档片段拼接为上下文字符串。"""
    return "\n\n".join(doc.page_content for doc in docs)


def to_messages(history: list) -> list[BaseMessage]:
    """将历史对话转换为 LangChain 消息列表。

    支持两种历史项表示：{"role": "user"/"assistant", "content": "..."}
    或 ("user"/"assistant", "...") 二元组；其余格式忽略。
    """
    messages = []
    for item in history or []:
        if isinstance(item, dict):
            role, content = item.get("role"), item.get("content")
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            role, content = item[0], item[1]
        else:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def build_rag_chain(vectorstore: Chroma, doc_id: str | None = None):
    """构建「检索 + 生成」链，返回 (chain, retriever)。

    doc_id 传入时限定只在某篇文档内检索（文档隔离）；为 None 则检索全部文档。
    chain 输入 {"input": 问题, "history": 历史消息列表}，输出回答文本；
    retriever 用于单独获取引用来源片段。
    """
    search_kwargs = {"k": config.RETRIEVER_K}
    if doc_id:
        search_kwargs["filter"] = {"doc_id": doc_id}
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    chain = (
        {
            "context": itemgetter("input") | retriever | _format_docs,
            "history": itemgetter("history"),
            "input": itemgetter("input"),
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    return chain, retriever


def answer_question(
    vectorstore: Chroma, question: str, history=None, doc_id: str | None = None
) -> tuple[str, list[Document]]:
    """对问题进行检索增强生成，返回 (回答文本, 引用来源片段列表)。"""
    chain, retriever = build_rag_chain(vectorstore, doc_id=doc_id)
    answer = chain.invoke({"input": question, "history": to_messages(history)})
    docs = retriever.invoke(question)
    return answer, docs


def stream_answer(vectorstore: Chroma, question: str, history=None, doc_id: str | None = None):
    """流式检索增强生成，返回 (回答 token 流, 引用来源片段列表)。"""
    chain, retriever = build_rag_chain(vectorstore, doc_id=doc_id)
    docs = retriever.invoke(question)
    return chain.stream({"input": question, "history": to_messages(history)}), docs
