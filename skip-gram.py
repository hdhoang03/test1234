import nltk
# nltk.download('punkt_tab')

import re
from collections import Counter
from nltk.tokenize import word_tokenize
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

# Bước 1: Tiền xử lý dữ liệu

def preprocess_text(text):
    """
    Chuyển đổi văn bản sang chữ thường, loại bỏ dấu câu và ký tự đặc biệt,
    tách từ và tạo danh sách từ.
    """
    # Chuyển về chữ thường
    text = text.lower()

    # Loại bỏ dấu câu và ký tự đặc biệt
    text = re.sub(r"[\W_]+", " ", text)

    # Tách từ (dùng nltk.word_tokenize cho đơn giản)
    tokens = word_tokenize(text)

    return tokens

# Tập dữ liệu mẫu tiếng Việt
text_data = """
Việt Nam là một đất nước xinh đẹp với nền văn hóa đa dạng.
Học sinh sinh viên thường thích học lập trình Python vì nó rất hữu ích.
"""

# Áp dụng tiền xử lý
tokens = preprocess_text(text_data)

# Tạo từ điển từ vựng
vocabulary = list(set(tokens))
vocab_size = len(vocabulary)
vocab_to_index = {word: idx for idx, word in enumerate(vocabulary)}
index_to_vocab = {idx: word for word, idx in vocab_to_index.items()}

print("Tokens:", tokens)
print("Vocabulary:", vocabulary)
print("Vocabulary Size:", vocab_size)
print("Word to Index Mapping:", vocab_to_index)
print("Index to Word Mapping:", index_to_vocab)

# Bước 2: Thiết kế mô hình Skip-gram
class SkipGramModel:
    def __init__(self, vocab_size, embedding_dim):
        """
        Khởi tạo mô hình Skip-gram với các tham số:
        - vocab_size: Kích thước từ vựng
        - embedding_dim: Kích thước vector nhúng
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Ma trận nhúng [|V| x d]
        self.embeddings = np.random.uniform(-1, 1, (vocab_size, embedding_dim))
        
        # Ma trận trọng số đầu ra [d x |V|]
        self.output_weights = np.random.uniform(-1, 1, (embedding_dim, vocab_size))

    def forward(self, center_word_idx):
        """
        Lan truyền tiến (forward pass):
        - center_word_idx: Chỉ số của từ trung tâm
        """
        # Lấy vector nhúng của từ trung tâm
        hidden_layer = self.embeddings[center_word_idx]

        # Tính xác suất của các từ trong ngữ cảnh bằng softmax
        output_layer = np.dot(hidden_layer, self.output_weights)
        probs = self.softmax(output_layer)

        return probs

    @staticmethod
    def softmax(x):
        """
        Hàm softmax để tính xác suất
        """
        exp_x = np.exp(x - np.max(x))  # Trừ max để tránh overflow
        return exp_x / np.sum(exp_x)

    def compute_loss(self, probs, target_idx):
        """
        Tính hàm mất mát cross-entropy
        - probs: Xác suất của các từ trong ngữ cảnh
        - target_idx: Chỉ số của từ mục tiêu
        """
        return -np.log(probs[target_idx])

# Khởi tạo mô hình
embedding_dim = 100  # Kích thước vector nhúng
model = SkipGramModel(vocab_size, embedding_dim)

# Bước 3: Huấn luyện mô hình
learning_rate = 0.01  # Tốc độ học
batch_size = 1  # Huấn luyện từng mẫu dữ liệu
epochs = 20  # Số lần lặp qua dữ liệu

# Tạo dữ liệu Skip-gram
def generate_skipgram_data(tokens, window_size=2):
    """
    Sinh dữ liệu Skip-gram từ danh sách từ
    """
    data = []
    for i, center_word in enumerate(tokens):
        context_indices = list(range(max(0, i - window_size), min(len(tokens), i + window_size + 1)))
        context_indices.remove(i)  # Loại bỏ từ trung tâm khỏi ngữ cảnh
        for context_idx in context_indices:
            data.append((center_word, tokens[context_idx]))
    return data

skipgram_data = generate_skipgram_data(tokens)

# Huấn luyện
epoch_losses = []  # Lưu trữ mất mát qua các epoch
for epoch in range(epochs):
    total_loss = 0
    for center_word, context_word in skipgram_data:
        center_idx = vocab_to_index[center_word]
        context_idx = vocab_to_index[context_word]

        # Lan truyền tiến
        probs = model.forward(center_idx)

        # Tính loss
        loss = model.compute_loss(probs, context_idx)
        total_loss += loss

        # Lan truyền ngược và cập nhật trọng số
        dl_doutput = probs
        dl_doutput[context_idx] -= 1  # Gradient của softmax + cross-entropy

        # Cập nhật ma trận trọng số đầu ra
        grad_output_weights = np.outer(model.embeddings[center_idx], dl_doutput)
        model.output_weights -= learning_rate * grad_output_weights

        # Cập nhật ma trận nhúng
        grad_embeddings = np.dot(model.output_weights, dl_doutput)
        model.embeddings[center_idx] -= learning_rate * grad_embeddings

    epoch_losses.append(total_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

# Vẽ biểu đồ mất mát qua các epoch
plt.plot(range(1, epochs + 1), epoch_losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss theo từng Epoch")
plt.show()

# Kiểm tra vector nhúng
word = "việt"
word_idx = vocab_to_index[word]
print(f"Vector nhúng cho từ '{word}':", model.embeddings[word_idx])

# Bước 4: Đánh giá vector nhúng
def cosine_similarity(vec1, vec2):
    """Tính cosine similarity giữa hai vector"""
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# Bộ từ mẫu để kiểm tra
word_pairs = [
    ("việt", "nam"),  # Kỳ vọng: tương đồng cao
    ("học", "lập trình"),  # Kỳ vọng: tương đồng trung bình
    ("học", "đẹp"),  # Kỳ vọng: tương đồng thấp
]

for word1, word2 in word_pairs:
    if word1 in vocab_to_index and word2 in vocab_to_index:
        vec1 = model.embeddings[vocab_to_index[word1]]
        vec2 = model.embeddings[vocab_to_index[word2]]
        similarity = cosine_similarity(vec1, vec2)
        print(f"Cosine similarity giữa '{word1}' và '{word2}': {similarity:.4f}")
    else:
        print(f"Một trong hai từ '{word1}' hoặc '{word2}' không có trong từ điển.")
