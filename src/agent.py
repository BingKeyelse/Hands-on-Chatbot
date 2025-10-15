# Import các thư viện cần thiết
from langchain.tools.retriever import create_retriever_tool  # Tạo công cụ tìm kiếm
from langchain_openai import ChatOpenAI  # Model ngôn ngữ OpenAI
from langchain.agents import AgentExecutor, create_react_agent  # Tạo và thực thi agent
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

# from dotenv import load_dotenv

# load_dotenv()

def get_retriever() -> EnsembleRetriever:
    """
    Tạo một ensemble retriever kết hợp vector search (Milvus) và BM25
    Returns:
        EnsembleRetriever: Retriever kết hợp với tỷ trọng:
            - 70% Milvus vector search (k=4 kết quả)
            - 30% BM25 text search (k=4 kết quả)
    Chú ý:
        - Yêu cầu Milvus server đang chạy tại localhost:19530
        - Collection 'database' phải tồn tại trong Milvus
        - BM25 được khởi tạo từ 100 document đầu tiên trong Milvus
    """

    # Kết nối với Milvus và tạo vector retriever
    vectorstore = connect_to_milvus('http://localhost:19530', 'database')

    # Retriever similarity
    milvus_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Lay lai data documents de lam input cho BM25
    documents = [Document(page_content=doc.page_content, metadata=doc.metadata)
                 for doc in vectorstore.similarity_search("", k=1000)]

    # Tạo BM25 retriever từ toàn bộ documents
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 4

    # Kết hợp hai retriever với tỷ trọng
    ensemble_retriever = EnsembleRetriever(
        retrievers=[milvus_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )
    return bm25_retriever

# --- KHỞI TẠO LLM (BẮT BUỘC PHẢI CACHE ĐỂ TRÁNH LOAD VRAM) ---
# Tên hàm đã được thay đổi để khớp với LLM
# @st.cache_resource(show_spinner="Đang tải Model LlamaCpp vào GPU P100 (Chỉ lần đầu)... 🚀")
# def get_local_llm() -> ChatLlamaCpp:
#     """
#     Tải mô hình ChatLlamaCpp vào VRAM GPU một lần duy nhất.
#     """
#     # LLM sẽ được giữ trong VRAM sau lần load đầu tiên
#     llm = ChatLlamaCpp(
#         model_path='vietnamese-llama2-7b-40gb.Q8_0.gguf',
#         temperature=0,
#         streaming=True,
#         n_ctx=4096,
#         # THAM SỐ CỰC KỲ QUAN TRỌNG CHO GPU P100
#         n_gpu_layers=33, # Đảm bảo toàn bộ model được offload lên P100 (cho Mistral 7B)
#         verbose=True
#     )
#     return llm

def get_local_llm() -> ChatOpenAI:
    """
    Kết nối tới mô hình LLM đã được expose qua API Server.
    Sử dụng ChatOpenAI để gọi API.
    """
    
    # 1. Khai báo URL của server API đã chạy:
    # Do bạn chạy trên 0.0.0.0:8001, bạn có thể dùng địa chỉ IP cụ thể của máy chủ
    # (ví dụ: 192.168.1.60) hoặc 0.0.0.0/localhost nếu gọi từ cùng máy.
    # Sử dụng IP mạng nội bộ của bạn (ví dụ: 192.168.1.60) để các máy khác có thể gọi.
    
    BASE_URL = "http://127.0.0.1:8001/v1"  # Hoặc "http://192.168.1.60:8001/v1"

    # 2. Khởi tạo ChatOpenAI client
    # llama_cpp.server mô phỏng API của OpenAI.
    llm = ChatOpenAI(
        model="vietnamese-llama2-7b-40gb.Q8_0.gguf", # Tên model tùy ý, miễn là server đang chạy
        openai_api_base=BASE_URL,
        openai_api_key="sk-not-required",  # Khóa API không cần thiết cho server nội bộ
        temperature=0,
        streaming=True,
        # Các tham số khác như n_ctx, n_gpu_layers KHÔNG được truyền ở đây, 
        # vì chúng đã được định cấu hình trên server (bằng lệnh --n_gpu_layers 33)
    )
    
    return llm


# Tạo công cụ tìm kiếm cho agent
def get_tools(retriever):
    """Tạo Tool thủ công từ Retriever để tránh lỗi tham số"""
    
    # Hàm search_function chỉ nhận một biến đầu vào duy nhất (query)
    def search_function(query: str):
        # Trả về kết quả search dưới dạng chuỗi
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])
    
    # Tạo Tool
    return [
        Tool(
            name="find",
            func=search_function,
            # THÊM HƯỚNG DẪN BẰNG TIẾNG VIỆT
            description="Công cụ này dùng để tìm kiếm thông tin từ cơ sở dữ liệu Milvus. Hãy sử dụng nó trước khi trả lời bất kỳ câu hỏi nào. Dùng ngôn ngữ Tiếng Việt khi đặt câu hỏi tìm kiếm (Action Input)."
        )
    ]

# @st.cache_resource(show_spinner="Khoi tao get_llm_and_agent lan dau 🚀")
# def get_llm_and_agent(_retriever, llm) -> AgentExecutor:
#     """
#     Khởi tạo Language Model và Agent với cấu hình cụ thể
#     Args:
#         _retriever: Retriever đã được cấu hình để tìm kiếm thông tin
#     Returns:
#         AgentExecutor: Agent đã được cấu hình với:
#             - Model: GPT-4
#             - Temperature: 0
#             - Streaming: Enabled
#             - Custom system prompt
#     Chú ý:
#         - Yêu cầu OPENAI_API_KEY đã được cấu hình
#         - Agent được thiết lập với tên "ChatchatAI"
#         - Sử dụng chat history để duy trì ngữ cảnh hội thoại
#     """
    
    
#     # Thiết lập prompt template cho agent
#     system = """You are an expert at AI. Your name is ChatchatAI.
    
#         You MUST always first use the provided 'find' tool to search for context 
#         and detailed information before attempting to answer ANY user question.
#         Only answer the question based on the facts you retrieve.
#         If the tool returns no relevant information, state that you cannot find the answer.
#         """
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{input}"),
#         MessagesPlaceholder(variable_name="agent_scratchpad"),
#     ])

#     # Tạo và trả về agent
#     agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
#     return AgentExecutor(agent=agent, tools=tools, verbose=False)
def get_llm_and_agent(_retriever, llm) -> AgentExecutor:
    
    # 1. KHỞI TẠO TOOLS
    tools = get_tools(_retriever) # <--- GỌI HÀM NÀY
    
    # 2. KHỞI TẠO MEMORY
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # 3. KHỞI TẠO AGENT
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=False, # Đã tắt verbose
        memory=memory, 
        handle_parsing_errors=True
    )

# # Khởi tạo retriever và agent
# retriever = get_retriever()
# llm= get_local_llm()
# agent_executor = get_llm_and_agent(retriever, llm)