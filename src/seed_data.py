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
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
from datetime import datetime
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility


def load_data_from_local(filename: str, directory: str):
    """
    Tự động đọc file văn bản theo định dạng bằng LangChain loader.
    Hỗ trợ: PDF, TXT, DOCX, CSV, MD
    """
    file_path = os.path.join(directory, filename)
    ext = filename.lower().split('.')[-1]

    # Chọn loader phù hợp
    if ext == 'pdf':
        loader = PyPDFLoader(file_path)
    elif ext == 'txt':
        loader = TextLoader(file_path, encoding='utf-8')
    elif ext == 'docx':
        loader = UnstructuredWordDocumentLoader(file_path)
    elif ext == 'csv':
        loader = CSVLoader(file_path)
    elif ext == 'md':
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        raise ValueError(f"❌ Định dạng file '{ext}' chưa được hỗ trợ.")

    # Load tài liệu (trả về list Document)
    documents = loader.load()
    return documents, filename, ext

def seed_milvus_local(URI_link: str, collection_name: str, filenames: list, directory: str) -> Milvus:
    # Khởi tạo model embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # # Kết nối Milvus
    # connections.connect("default", uri=URI_link)

    # # Tạo schema nếu chưa có collection
    # if not utility.has_collection(collection_name):
    #     print(f"📦 Collection '{collection_name}' chưa tồn tại — tạo mới...")

    #     fields = [
    #         FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, max_length=64),  # đổi từ id → pk
    #         FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
    #         FieldSchema(name="doc_name", dtype=DataType.VARCHAR, max_length=255),
    #         FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=50),
    #         FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
    #         FieldSchema(name="date_update", dtype=DataType.VARCHAR, max_length=20)
    #     ]

    #     schema = CollectionSchema(fields=fields, description="Local text embedding storage")
    #     collection = Collection(name=collection_name, schema=schema)
    #     print(f"✅ Đã tạo collection '{collection_name}' thành công.")
    # else:
    #     collection = Collection(collection_name)
    #     print(f"✅ Collection '{collection_name}' đã tồn tại.")

    # Khởi tạo Milvus vectorstore (không ép consistency_level)
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,
        drop_old=False
    )

    # Lấy thời gian hiện tại
    current_time = datetime.now().strftime("%Y-%m-%d")

    # Xử lý từng file trong danh sách
    for filename in filenames:
        local_data, doc_name, type_doc = load_data_from_local(filename, directory)

        # ✂️ Chia nhỏ văn bản
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        all_splits = text_splitter.split_documents(local_data)

        # Tạo danh sách Document
        documents = [
            Document(
                page_content=doc.page_content if hasattr(doc, "page_content") else doc.get("page_content", ""),
                metadata={
                    'doc_name': doc_name or '',
                    'content_type': type_doc or '',
                    'source': directory or '',
                    'date_update': current_time
                }
            )
            for doc in all_splits
        ]

        uuids = [str(uuid4()) for _ in range(len(documents))]

        # # ⚙️ Kiểm tra xem doc_name + content_type đã tồn tại chưa bằng query
        # try:
        #     collection.load()
        #     expr = f'doc_name == "{doc_name}" and content_type == "{type_doc}"'
        #     result = collection.query(expr=expr, output_fields=["id"])
        # except Exception as e:
        #     print(f"⚠️ Không thể query collection: {e}")
        #     result = []

        # if len(result) > 0:
        #     print(f"🔁 Phát hiện tài liệu trùng '{doc_name}' ({type_doc}) — xóa và cập nhật lại...")
        #     collection.delete(expr)
        #     collection.flush()

        # 🧩 Thêm mới hoặc cập nhật document
        vectorstore.add_documents(documents=documents, ids=uuids)
        print(f"✅ Đã lưu '{doc_name}' ({type_doc}) vào collection '{collection_name}'")

    return vectorstore