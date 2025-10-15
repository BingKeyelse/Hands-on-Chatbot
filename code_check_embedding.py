# Cần cài đặt: pip install sentence-transformers scikit-learn numpy

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import torch

# 1. KHỞI TẠO MÔ HÌNH EMBEDDING
# Sử dụng mô hình e5-large-v2 như yêu cầu của bạn
print("Đang tải mô hình intfloat/e5-large-v2...")
try:
    # SentenceTransformer sẽ tự động tải mô hình từ Hugging Face
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("intfloat/e5-large-v2").to(device=device)
    print("✅ Tải mô hình hoàn tất.")
except Exception as e:
    print(f"❌ Lỗi khi tải mô hình: {e}")
    print("Vui lòng đảm bảo bạn có kết nối mạng và thư viện đã được cài đặt.")
    exit()


def calculate_similarity(text_a: str, text_b: str) -> float:
    """
    Tính toán độ tương đồng Cosine giữa hai đoạn văn bản bằng mô hình E5-large-v2.
    LƯU Ý QUAN TRỌNG: Sử dụng tiền tố 'query: ' cho cả hai đoạn văn bản để tối ưu hóa
    hiệu suất và khả năng phân biệt điểm số của mô hình E5.
    """
    # 2. MÃ HÓA (ENCODE) CẢ HAI ĐOẠN VĂN BẢN CÙNG MỘT LÚC
    # Thêm tiền tố 'query: ' cho cả hai câu
    sentences = [f"query: {text_a}", f"query: {text_b}"]
    
    # Mã hóa với show_progress_bar=False để giữ giao diện console gọn gàng
    embeddings: np.ndarray = model.encode(sentences, show_progress_bar=False)
    
    # 3. CHUẨN BỊ VECTOR CHO TÍNH TOÁN
    # embeddings[0] là vector của text_a
    # embeddings[1] là vector của text_b
    vector_a = embeddings[0].reshape(1, -1) # Phải reshape thành (1, N) cho cosine_similarity
    vector_b = embeddings[1].reshape(1, -1)
    
    # 4. TÍNH TOÁN ĐỘ TƯƠNG ĐỒNG COSINE
    # Kết quả là một mảng 2D, ta chỉ cần giá trị [0][0]
    similarity_score = cosine_similarity(vector_a, vector_b)[0][0]
    
    return float(similarity_score)


if __name__ == "__main__":
    # --- ĐOẠN VĂN BẢN ĐỂ KIỂM TRA ---
    
    # 1. Độ tương đồng CAO (Nội dung gần như nhau)
    text_1_a = "The chef created a wonderful dish to serve the diners."
    text_1_b = "The cook made an exquisite dinner."
    
    # 2. Độ tương đồng TRUNG BÌNH (Chủ đề tương tự)
    text_2_a = "She possesses strong knowledge of data science."
    text_2_b = "He is passionate about studying deep learning."
    
    # 3. Độ tương đồng THẤP (Hoàn toàn khác biệt)
    text_3_a = "It is raining lightly in Paris this morning."
    text_3_b = "I plan to pick up some fresh vegetables from the store."

    
    print("\n--- KẾT QUẢ KIỂM TRA ĐỘ TƯƠNG ĐỒNG ---")
    
    # Ví dụ 1: Rất Cao
    score_1 = calculate_similarity(text_1_a, text_1_b)
    print(f"\n[Ví dụ 1: TƯƠNG ĐỒNG CAO]")
    print(f"Văn bản A: {text_1_a}")
    print(f"Văn bản B: {text_1_b}")
    print(f"-> Điểm tương đồng Cosine: {score_1:.4f}")
    
    # Ví dụ 2: Trung Bình
    score_2 = calculate_similarity(text_2_a, text_2_b)
    print(f"\n[Ví dụ 2: TƯƠNG ĐỒNG TRUNG BÌNH]")
    print(f"Văn bản A: {text_2_a}")
    print(f"Văn bản B: {text_2_b}")
    print(f"-> Điểm tương đồng Cosine: {score_2:.4f}")

    # Ví dụ 3: Thấp
    score_3 = calculate_similarity(text_3_a, text_3_b)
    print(f"\n[Ví dụ 3: TƯƠNG ĐỒNG THẤP]")
    print(f"Văn bản A: {text_3_a}")
    print(f"Văn bản B: {text_3_b}")
    print(f"-> Điểm tương đồng Cosine: {score_3:.4f}")
