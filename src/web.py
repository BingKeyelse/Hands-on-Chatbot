import streamlit as st  # Thư viện tạo giao diện web
import time
# from langchain.memory import StreamlitChatMessageHistory  # Lưu lịch sử chat
from langchain_community.chat_message_histories import StreamlitChatMessageHistory # Lưu lịch sử chat
import os
# from langchain.callbacks import StreamlitCallbackHandler  # Hiển thị kết quả realtime
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler # Hiển thị kết quả realtime

from seed_data import seed_milvus_local, seed_milvus_live
from agent import get_retriever, get_llm_and_agent, get_local_llm  # Khởi tạo AI

# streamlit run web.py
# streamlit run web.py --server.address 0.0.0.0 --server.port 8501



# === THIẾT LẬP GIAO DIỆN TRANG WEB ===
def setup_page():
    """
    Cấu hình trang web cơ bản
    """
    st.set_page_config(
        page_title="AI Assistant",  # Tiêu đề tab trình duyệt
        page_icon="💬",  # Icon tab
        layout="wide"  # Giao diện rộng
    )

# === KHỞI TẠO ỨNG DỤNG ===
def initialize_app():

    setup_page()  # Thiết lập giao diện

# === THANH CÔNG CỤ BÊN TRÁI ===
def setup_sidebar():
    """
    Tạo thanh công cụ bên trái với các tùy chọn:
    1. Chọn nguồn dữ liệu (File hoặc URL)
    2. Nhập thông tin file/URL
    3. Nút tải/crawl dữ liệu
    """
    with st.sidebar:
        st.title("⚙️ Cấu hình")

        # Chọn nguồn dữ liệu
        data_source = st.radio(
            "Chọn nguồn dữ liệu:",
            ["File Local", "URL trực tiếp"]
        )
        
        # Xử lý tùy theo lựa chọn
        if data_source == "File Local":
            handle_local_file()
        else:
            handle_url_input()


def handle_local_file():
    """
    Xử lý khi người dùng chọn tải file:
    1. Nhập tên file và thư mục
    2. Tải dữ liệu khi nhấn nút
    """
    # Upload file
    st.info("💡 Hãy chọn file (PDF, TXT, DOCX, CSV, MD) để upload.")

    uploaded_files = st.file_uploader("Chọn file để upload", 
                                        accept_multiple_files=True,
                                        type=["pdf", "txt", "docx", "csv", "md"]
                                        )
    if uploaded_files:
        data=[]
        for uploaded_file in uploaded_files:
            save_path = os.path.join('Data_restore', uploaded_file.name)
            data.append(uploaded_file.name)

            # Write file into folder Data_restore to read after
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        if st.button("Handle data"):
            with st.spinner("Handling..."):
                seed_milvus_local('http://localhost:19530','database', data, 'Data_restore')
            st.success("Done!")

def handle_url_input():
    """
    Xử lý khi người dùng chọn crawl URL:
    1. Nhập URL cần crawl
    2. Bắt đầu crawl khi nhấn nút
    """
    url = st.text_input("Nhập URL:", "https://www.stack-ai.com/docs")
    if st.button("Crawl dữ liệu"):
        with st.spinner("Đang crawl dữ liệu..."):
            seed_milvus_live(url, 'http://localhost:19530', 'database')
        st.success("Đã crawl dữ liệu thành công!")

# === GIAO DIỆN CHAT CHÍNH ===
# === GIAO DIỆN CHAT CHÍNH ===
def setup_chat_interface():
    st.title("💬 AI Assistant")
    

    st.caption("🚀 Trợ lý AI được hỗ trợ bởi LangChain và Ollama")
    
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}
        ]
        msgs.add_ai_message("Tôi có thể giúp gì cho bạn?")

    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "human"
        st.chat_message(role).write(msg["content"])

    return msgs

# === XỬ LÝ TIN NHẮN NGƯỜI DÙNG ===
# === XỬ LÝ TIN NHẮN NGƯỜI DÙNG ===
def handle_user_input(msgs, agent_executor):
    """
    Xử lý khi người dùng gửi tin nhắn:
    1. Hiển thị tin nhắn người dùng
    2. Gọi AI xử lý và trả lời
    3. Lưu vào lịch sử chat
    """
    if prompt := st.chat_input("Hãy hỏi tôi bất cứ điều gì về Stack AI!"):
        # Lưu và hiển thị tin nhắn người dùng
        st.session_state.messages.append({"role": "human", "content": prompt})
        st.chat_message("human").write(prompt)
        msgs.add_user_message(prompt)

        # Xử lý và hiển thị câu trả lời
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            
            # Lấy lịch sử chat
            chat_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages[:-1]
            ]

            # Gọi AI xử lý
            response = agent_executor.invoke(
                {
                    "input": prompt,
                    "chat_history": chat_history
                },
                {"callbacks": [st_callback]}
            )

            # Lưu và hiển thị câu trả lời
            output = response["output"]
            st.session_state.messages.append({"role": "assistant", "content": output})
            msgs.add_ai_message(output)
            st.write(output)

# === HÀM CHÍNH ===
def main():
    """
    Hàm chính điều khiển luồng chương trình:
    1. Khởi tạo ứng dụng
    2. Tạo giao diện
    3. Xử lý tương tác người dùng
    """
    initialize_app()
    setup_sidebar()
    
    # 💡 SỬA: CHỈ GỌI setup_chat_interface(), KHÔNG GÁN VÀ SỬ DỤNG MSGS NỮA
    msgs = setup_chat_interface()

    retriever = get_retriever()
    llm_instance = get_local_llm()
    # Agent Executor đã có Memory riêng (ConversationBufferMemory)
    agent_executor = get_llm_and_agent(retriever, llm_instance) 

    # 💡 SỬA: CHỈ GỌI HÀM VỚI agent_executor
    handle_user_input(msgs, agent_executor) 

def test_retriever_query(query: str):
    """
    Hàm test truy vấn trực tiếp Retriever
    """
    print(f"\n--- 🔎 ĐANG TRUY VẤN: '{query}' ---")
    
    try:
        # Lấy retriever đã được cấu hình (EnsembleRetriever)
        retriever = get_retriever()
        
        # ⚠️ SỬ DỤNG PHƯƠNG THỨC invoke()
        # Đối với LangChain v0.2, invoke() là cách được khuyến nghị
        results = retriever.invoke(query)
        
        # In kết quả
        print(f"✅ Đã tìm thấy {len(results)} tài liệu:")
        for i, doc in enumerate(results):
            # In nội dung ngắn gọn và nguồn
            content_snippet = doc.page_content[:200] + "..."
            name = doc.metadata.get('doc_name', 'N/A')
            # source = doc.metadata.get('source', 'N/A')
            print(f"--- Document {i+1} ---")
            print(f"Nguồn: {name}")
            # print(f"Nội dung: {content_snippet}")
            print("-" * 15)

    except Exception as e:
        print(f"❌ Lỗi xảy ra khi truy vấn Retriever: {e}")
        print("Vui lòng đảm bảo Milvus server đang chạy tại localhost:19530")

# Chạy ứng dụng
if __name__ == "__main__":
    main() 
   
