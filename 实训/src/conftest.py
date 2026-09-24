"""pytest 配置：确保项目根目录可被 `import config / rag / document_loader`。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
