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

import nltk
nltk.download('averaged_perceptron_tagger_eng')

from craw_data import load_data_from_local, crawl_web

# Khởi tạo model embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def seed_milvus_local(URI_link: str, collection_name: str, filenames: list, directory: str) -> Milvus:
    # --- Kết nối tới Milvus ---
    print("🔗 Kết nối tới Milvus...")
    connections.connect("default", uri=URI_link)
    
    if not utility.has_collection(collection_name):
        print(f"📦 Collection '{collection_name}' chưa tồn tại — tạo mới...")
        vectorstore = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": URI_link},
            collection_name=collection_name,
            drop_old=False
        )
    else:
        print(f"✅ Collection '{collection_name}' đã tồn tại.")
        vectorstore = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": URI_link},
            collection_name=collection_name,
            drop_old=False
        )

    # Truy cập trực tiếp collection trong Milvus (để query / delete)
    collection = Collection(collection_name)


    # Lấy thời gian hiện tại
    current_time = datetime.now().strftime("%Y-%m-%d")

    # Xử lý từng file trong danh sách
    for filename in filenames:
        local_data, doc_name, type_doc = load_data_from_local(filename, directory)
        print(doc_name)

        # ✂️ Chia nhỏ văn bản
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
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

        # 🔍 Kiểm tra xem doc_name + content_type có tồn tại không
        expr = f'doc_name == "{doc_name}" and content_type == "{type_doc}"'
        try:
            collection.load()
            result = collection.query(expr=expr, output_fields=["pk"])
        except Exception as e:
            print(f"⚠️ Query lỗi: {e}")
            result = []

        # Nếu tồn tại → xóa bản cũ trước khi thêm
        if result:
            print(f"🔁 Tài liệu '{doc_name}' ({type_doc}) đã tồn tại — xóa dữ liệu cũ...")
            collection.delete(expr)
            collection.flush()
        
        vectorstore.add_documents(documents=documents, ids=uuids)
        print(f"✅ Đã lưu '{doc_name}' ({type_doc}) vào collection '{collection_name}'")

    return vectorstore

def seed_milvus_live(URL: str, URI_link: str, collection_name: str) -> Milvus:

    # --- Kết nối tới Milvus ---
    print("🔗 Kết nối tới Milvus...")
    connections.connect("default", uri=URI_link)
    
    if not utility.has_collection(collection_name):
        print(f"📦 Collection '{collection_name}' chưa tồn tại — tạo mới...")
        vectorstore = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": URI_link},
            collection_name=collection_name,
            drop_old=False
        )
    else:
        print(f"✅ Collection '{collection_name}' đã tồn tại.")
        vectorstore = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": URI_link},
            collection_name=collection_name,
            drop_old=False
        )

    # Truy cập trực tiếp collection trong Milvus (để query / delete)
    collection = Collection(collection_name)


    # Lấy thời gian hiện tại
    current_time = datetime.now().strftime("%Y-%m-%d")

    documents = crawl_web(URL)

    # Chia nhỏ văn bản thành các đoạn 10000 ký tự, với 500 ký tự chồng lấp
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    documents = text_splitter.split_documents(documents)
    print('length_all_splits: ', len(documents))  # In số lượng đoạn văn bản sau khi chia

    # --- Cập nhật metadata cho mỗi document ---
    for doc in documents:
        doc_name = doc.metadata.get('title') or 'untitled'
        type_doc = doc.metadata.get('content_type') or 'text_web'

        doc.metadata = {
            'doc_name': doc_name,
            'content_type': type_doc,
            'source': doc.metadata.get('source') or '',
            'date_update': current_time
        }
        

        # --- Kiểm tra nếu tồn tại document trùng thì xóa ---
        # expr = f'doc_name == "{doc_name}" and content_type == "{type_doc}"'
        expr = f'doc_name == "{doc_name}" and source == "{doc.metadata.get("source")}"'
        try:
            collection.load()
            result = collection.query(expr=expr, output_fields=["pk"])
        except Exception as e:
            print(f"⚠️ Query lỗi: {e}")
            result = []

        if result:
            print(f"🔁 Tài liệu '{doc_name}' ({type_doc}) đã tồn tại — xóa dữ liệu cũ...")
            collection.delete(expr)
            collection.flush()

    # --- Tạo UUID và thêm documents vào Milvus ---
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vectorstore.add_documents(documents=documents, ids=uuids)
    print(f"✅ Đã lưu {len(documents)} đoạn vào collection '{collection_name}'")
    return vectorstore

def connect_to_milvus(URI_link: str, collection_name: str) -> Milvus:
    """
    Hàm kết nối đến collection có sẵn trong Milvus
    Args:
        URI_link (str): Đường dẫn kết nối đến Milvus
        collection_name (str): Tên collection cần kết nối
    Returns:
        Milvus: Đối tượng Milvus đã được kết nối, sẵn sàng để truy vấn
    Chú ý:
        - Không tạo collection mới hoặc xóa dữ liệu cũ
        - Sử dụng model 'text-embedding-3-large' cho việc tạo embeddings khi truy vấn
    """
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,
    )
    return vectorstore