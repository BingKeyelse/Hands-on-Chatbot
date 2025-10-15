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

from dotenv import load_dotenv

load_dotenv()

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

def get_local_llm() -> ChatOpenAI:
    """
    Kết nối tới mô hình LLM đã được expose qua API Server.
    Sử dụng ChatOpenAI để gọi API.
    """
    # 1. Khai báo URL của server API đã chạy:
    # Do bạn chạy trên 0.0.0.0:8001, bạn có thể dùng địa chỉ IP cụ thể của máy chủ
    # (ví dụ: 192.168.1.60) hoặc 0.0.0.0/localhost nếu gọi từ cùng máy.
    # Sử dụng IP mạng nội bộ của bạn (ví dụ: 192.168.1.60) để các máy khác có thể gọi.
    
    BASE_URL = "http://192.168.1.83:8001/v1" # "http://127.0.0.1:8001/v1"  Hoặc "http://192.168.1.83:8001/v1"
    BASE_URL_OLLAMA = "http://192.168.1.83:11434/v1" 

    # 2. Khởi tạo ChatOpenAI client
    # llama_cpp.server mô phỏng API của OpenAI.
    def get_ollama_llm() -> ChatOpenAI:
        try:
            llm = ChatOpenAI(
                model="deepseek-coder:8b", 
                openai_api_base=BASE_URL_OLLAMA,
                openai_api_key="sk-not-required",
                temperature=0.7,
                streaming=True,
                max_tokens=2048,
            )
            # Đảm bảo có lệnh return này!
            return llm 
        except Exception as e:
            # Nếu có lỗi ở đây, hàm sẽ thoát và trả về None.
            # Nên xử lý lỗi (ví dụ: in ra thông báo lỗi) nhưng không return None.
            print(f"Lỗi khởi tạo LLM: {e}")
            # Nếu không thể khởi tạo, bạn có thể return None, nhưng hãy xử lý nó sau này
            # Ví dụ: raise e
            return None # <-- Nếu bạn có dòng này, code gọi phải kiểm tra nó.


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

if __name__ == "__main__":
    # Khởi tạo retriever và agent
    # retriever = get_retriever()
    llm= get_local_llm()
    # agent_executor = get_llm_and_agent(retriever, llm)
    # ----------------------------------------------------
    # Phần code để test LLM với câu hỏi của bạn (Sử dụng .stream())
    # ----------------------------------------------------
    question = "Cho tôi một bài thơ hay về núi"
    
    # 1. Tạo một tin nhắn (message) từ người dùng
    from langchain.schema import HumanMessage, SystemMessage

    messages = [
    SystemMessage(content="Bạn là một nhà thơ tài năng và chuyên nghiệp. Hãy trả lời các yêu cầu bằng một bài thơ độc đáo."),
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