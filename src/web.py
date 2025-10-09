import streamlit as st  # Thư viện tạo giao diện web
import time
from langchain.memory import StreamlitChatMessageHistory  # Lưu lịch sử chat
import os
from seed_data import seed_milvus_local

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
    """
    Khởi tạo các cài đặt cần thiết:
    - Đọc file .env chứa API key
    - Cấu hình trang web
    """
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

            # Ghi dữ liệu ra file
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
            pass
            # seed_milvus_live(url, 'http://localhost:19530', 'data_test_live_v2', 'stack-ai')
        st.success("Đã crawl dữ liệu thành công!")

# === GIAO DIỆN CHAT CHÍNH ===
def setup_chat_interface():
    """
    Tạo giao diện chat chính:
    1. Hiển thị tiêu đề
    2. Khởi tạo lịch sử chat
    3. Hiển thị các tin nhắn
    """
    st.title("💬 AI Assistant")
    st.caption("🚀 Trợ lý AI được hỗ trợ bởi LangChain và OpenAI")

    # Khởi tạo bộ nhớ chat
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    
    # Tạo tin nhắn chào mừng nếu là chat mới
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}
        ]
        msgs.add_ai_message("Tôi có thể giúp gì cho bạn?")

    # Hiển thị lịch sử chat
    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "human"
        st.chat_message(role).write(msg["content"])

    return msgs

# === XỬ LÝ TIN NHẮN NGƯỜI DÙNG ===
def handle_user_input(msgs):
    """
    Xử lý khi người dùng gửi tin nhắn:
    1. Hiển thị tin nhắn người dùng
    2. Gọi AI xử lý và trả lời
    3. Lưu vào lịch sử chat
    """
    if prompt := st.chat_input("Hãy hỏi tôi bất cứ điều gì về Stack AI!"):
        # Lưu và hiển thị tin nhắn người dùng
        st.session_state.messages.append({"role": "human", "content": prompt})
        st.chat_message("human").write(prompt) # cho vai trò và prompt mình nhập
        msgs.add_user_message(prompt)

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
    # msgs = setup_chat_interface()

    # # Xử lý chat
    # handle_user_input(msgs)

# Chạy ứng dụng
if __name__ == "__main__":
    main() 