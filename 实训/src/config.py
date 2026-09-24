"""全局配置：加载环境变量，统一读取路径与模型名。"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 国内镜像加速 HuggingFace 模型下载（嵌入模型）。
# 必须在任何会导入 huggingface_hub 的库（langchain_community / sentence-transformers）
# 之前设置，否则其会固化默认端点 https://huggingface.co（国内被墙）。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 项目根目录（src/ 的上一级，存放 .env、chroma_db/、documents.json）
BASE_DIR = Path(__file__).resolve().parent.parent

# 加载 .env（若存在）
load_dotenv(BASE_DIR / ".env")

# DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# LLM 调用参数：温度、超时（秒）与失败重试次数（应对限流/超时）
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# 嵌入模型
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

# 文本切分参数
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# 检索返回片段数
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "4"))

# 向量库持久化目录
CHROMA_DIR = str(BASE_DIR / "chroma_db")

# Chroma 集合名：所有文档统一存于该集合，靠 doc_id 元数据隔离
CHROMA_COLLECTION = "documents"
