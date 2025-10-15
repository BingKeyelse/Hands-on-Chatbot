# Import các thư viện cần thiết
from langchain.tools.retriever import create_retriever_tool  # Tạo công cụ tìm kiếm
from langchain_openai import ChatOpenAI  # Model ngôn ngữ OpenAI
from langchain.agents import AgentExecutor, create_react_agent, create_openai_functions_agent # Tạo và thực thi agent
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Xử lý prompt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate # Xử lý prompt
from seed_data import connect_to_milvus  # Kết nối với Milvus
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler  # Xử lý callback cho Streamlit
from langchain_community.chat_message_histories import StreamlitChatMessageHistory  # Lưu trữ lịch sử chat
from langchain.retrievers import EnsembleRetriever  # Kết hợp nhiều retriever
from langchain_community.retrievers import BM25Retriever  # Retriever dựa trên BM25
from langchain_core.documents import Document  # Lớp Document
from pymilvus import Collection
from langchain_community.chat_models import ChatLlamaCpp
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory 
from langchain.agents import initialize_agent, AgentType

from langchain.tools import Tool # <--- Thêm Tool thủ công
from langchain.agents import initialize_agent, AgentType 
from langchain.memory import ConversationBufferMemory 
import streamlit as st # Bắt buộc phải import Streamlit

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List, Any
import warnings

from dotenv import load_dotenv

load_dotenv()

def get_retriever() -> EnsembleRetriever:
    try:
        print('Tôi được gọi rồi')
        # Kết nối với Milvus và tạo vector retriever
        vectorstore = connect_to_milvus('http://localhost:19530', 'database')

        # Retriever similarity
        milvus_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        try:
            documents = vectorstore.similarity_search("", k=1000) 
        except Exception as e:
            print(f"Lỗi khi lấy documents từ Milvus: {e}")
            documents = []

        # KIỂM TRA LỖI (Giải quyết lỗi ValueError: not enough values to unpack)
        if not documents:
            # Nếu không có documents, BM25 không thể khởi tạo. Trả về Milvus retriever một mình.
            print("CẢNH BÁO: Không lấy được documents từ Milvus. Chỉ sử dụng MilvusRetriever.")
            return milvus_retriever 

        # 4. Tạo BM25 retriever từ documents
        # documents đã là danh sách các đối tượng Document, nên có thể truyền trực tiếp
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 4 # Giới hạn kết quả trả về của BM25

        # 5. Kết hợp hai retriever với tỷ trọng
        ensemble_retriever = EnsembleRetriever(
            retrievers=[milvus_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
        return ensemble_retriever
    
    except Exception as e:
        print(f"Lỗi khi khởi tạo retriever: {str(e)}")
        # Trả về retriever với document mặc định nếu có lỗi
        default_doc = [
            Document(
                page_content="Có lỗi xảy ra khi kết nối database. Vui lòng thử lại sau.",
                metadata={"source": "error"}
            )
        ]
        return BM25Retriever.from_documents(default_doc)

def get_local_llm() -> ChatOpenAI:
    # BASE_URL = "http://192.168.1.83:8001/v1" # "http://127.0.0.1:8001/v1"  Hoặc "http://192.168.1.83:8001/v1"
    BASE_URL_OLLAMA = "http://192.168.1.83:11434/v1" 

    try:
        llm = ChatOpenAI(
            model="deepseek-r1:8b", 
            openai_api_base=BASE_URL_OLLAMA,
            openai_api_key="sk-not-required",
            temperature=0.7,
            streaming=True,
            max_tokens=2048,
        )
        # Đảm bảo có lệnh return này!
        return llm 
    except Exception as e:
        print(f"Lỗi khởi tạo LLM: {e}")
        return None # <-- Nếu bạn có dòng này, code gọi phải kiểm tra nó.

def get_llm_and_agent(_retriever, llm) -> AgentExecutor:
    # Tạo công cụ tìm kiếm cho agent
    tool = create_retriever_tool(
        get_retriever(),
        "find", # Tên tool: find
        # ⚠️ ĐÃ SỬA: BẮT BUỘC DÙNG TOOL trong mô tả để ép Agent gọi hàm
        "Công cụ này dùng để tìm kiếm thông tin từ cơ sở dữ liệu Milvus. " \
        "HÃY LUÔN LUÔN VÀ BẮT BUỘC SỬ DỤNG CÔNG CỤ NÀY TRƯỚC KHI TRẢ LỜI MỌI CÂU HỎI thực tế hoặc chuyên ngành. " \
        "Dùng ngôn ngữ Tiếng Việt khi đặt câu hỏi tìm kiếm (Action Input)."
    )
    
    tools = [tool]
    # Khởi tạo ChatOpenAI với chế độ streaming
    llm =llm
    
    # Thiết lập prompt template
    system = """Bạn là một trợ lý ảo chuyên nghiệp, lịch sự và đáng tin cậy tên ChatchatAI. 
    Nhiệm vụ cốt lõi của bạn là trả lời các câu hỏi dựa trên kiến thức được cung cấp từ công cụ 'find'. 
    BẠN PHẢI SỬ DỤNG CÔNG CỤ 'find' CHO MỌI CÂU HỎI MỚI (trừ khi đó chỉ là lời chào hỏi hoặc cảm ơn đơn thuần)."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Tạo agent
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)

def test_llm():
    llm= get_local_llm()
    question = "Cho tôi một bài thơ hay về núi"

    # 1. Tạo một tin nhắn (message) từ người dùng
    from langchain.schema import HumanMessage, SystemMessage

    messages = [
    SystemMessage(content="Bạn là một nhà thơ tài năng và chuyên nghiệp. Hãy trả lời các yêu cầu bằng một bài thơ độc đáo bằng tiếng Việt"),
    HumanMessage(content=question)
]
    
    print(f"Câu hỏi: {question}\n")
    print("--- Phản hồi từ LLM (Streaming) ---")
    
    # 2. Gọi mô hình LLM sử dụng .stream()
    try:
        # Sử dụng .stream() để nhận generator
        response_stream = llm.stream(messages)
        
        # 3. Lặp qua generator và in từng phần
        for chunk in response_stream:
            # chunk.content chứa phần văn bản nhỏ
            print(chunk.content, end="", flush=True)
            
        # 4. In một dòng mới sau khi stream kết thúc
        print() 
        
    except Exception as e:
        print(f"\nĐã xảy ra lỗi khi gọi LLM: {e}")
        print("Vui lòng kiểm tra lại server API của bạn (địa chỉ IP và cổng).")
    
    print("-----------------------------------")


if __name__ == "__main__":
    # Khởi tạo retriever và agent
    # retriever = get_retriever()
    # llm= get_local_llm()
    # agent_executor = get_llm_and_agent(retriever, llm)
    test_llm()

