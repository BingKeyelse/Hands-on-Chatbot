import os
import json
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain.schema import Document
from dotenv import load_dotenv
from uuid import uuid4
# from crawl import crawl_web
from langchain.embeddings import HuggingFaceEmbeddings

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader
)
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import RecursiveUrlLoader, WebBaseLoader
from bs4 import BeautifulSoup

import nltk
nltk.download('averaged_perceptron_tagger_eng')

def load_data_from_local(filename: str, directory: str):
    """
    Tự động đọc file văn bản theo định dạng bằng LangChain loader.
    Hỗ trợ: PDF, TXT, DOCX, CSV, MD
    """
    file_path = os.path.join(directory, filename)
    ext = filename.lower().split('.')[-1]

    # Chọn loader phù hợp
    if ext == 'pdf':
        # loader = PyPDFLoader(file_path)
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # 🧠 Gộp toàn bộ nội dung các trang lại thành 1 document duy nhất
        merged_text = "\n".join([d.page_content for d in docs])
        documents = [
            Document(
                page_content=merged_text,
                metadata={"source": file_path}
            )
        ]
    elif ext == 'txt':
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
    elif ext == 'docx':
        loader = UnstructuredWordDocumentLoader(file_path)
        documents = loader.load()
    elif ext == 'csv':
        loader = CSVLoader(file_path)
        documents = loader.load()
    elif ext == 'md':
        loader = UnstructuredMarkdownLoader(file_path)
        documents = loader.load()
    else:
        raise ValueError(f"❌ Định dạng file '{ext}' chưa được hỗ trợ.")

    # Load tài liệu (trả về list Document)
    # documents = loader.load()
    return documents, filename, ext

def bs4_extractor(html: str) -> str:
    """
    Hàm trích xuất và làm sạch nội dung từ HTML
    Args:
        html: Chuỗi HTML cần xử lý
    Returns:
        str: Văn bản đã được làm sạch, loại bỏ các thẻ HTML và khoảng trắng thừa
    """
    soup = BeautifulSoup(html, "html.parser")  # Phân tích cú pháp HTML
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()  # Xóa khoảng trắng và dòng trống thừa

def crawl_web(url_data):
    """
    Hàm crawl dữ liệu từ URL với chế độ đệ quy
    Args:
        url_data (str): URL gốc để bắt đầu crawl
    Returns:
        list: Danh sách các Document object, mỗi object chứa nội dung đã được chia nhỏ
              và metadata tương ứng
    """
    # Tạo loader với độ sâu tối đa là 4 cấp
    loader = RecursiveUrlLoader(url=url_data, extractor=bs4_extractor, max_depth=4)
    docs = loader.load()  # Tải nội dung
    print('length: ', len(docs))  # In số lượng tài liệu đã tải
    
    return docs