import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

print("Đang tải dữ liệu và chuẩn bị vẽ biểu đồ...")
# Đọc file và dọn sạch dữ liệu trống để không bị lỗi
df = pd.read_csv('financial_anomaly_data.csv').dropna()

# Chỉ lấy cột Amount để vẽ biểu đồ cho rõ ràng sự đột biến về số tiền
X = df[['Amount']]

# ==========================================
# 1. CHẠY ISOLATION FOREST
# ==========================================
start_if = time.time()
model_if = IsolationForest(contamination=0.01, random_state=42)
df['Nhan_IF'] = model_if.fit_predict(X)
time_if = time.time() - start_if
gian_lan_if = len(df[df['Nhan_IF'] == -1])

# ==========================================
# 2. CHẠY LOCAL OUTLIER FACTOR (LOF)
# ==========================================
print("Đang tính toán LOF... (Vui lòng chờ khoảng 1-2 phút)")
start_lof = time.time()
model_lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
df['Nhan_LOF'] = model_lof.fit_predict(X)
time_lof = time.time() - start_lof
gian_lan_lof = len(df[df['Nhan_LOF'] == -1])

# ==========================================
# 3. VẼ BIỂU ĐỒ SO SÁNH TRỰC QUAN
# ==========================================
print("Đang xuất ảnh biểu đồ...")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Khung 1: Isolation Forest (Màu đỏ)
axes[0].scatter(df.index[df['Nhan_IF'] == 1], df['Amount'][df['Nhan_IF'] == 1], color='blue', alpha=0.3, s=10, label='Bình thường')
axes[0].scatter(df.index[df['Nhan_IF'] == -1], df['Amount'][df['Nhan_IF'] == -1], color='red', marker='X', s=50, label='Gian lận (Dị biệt)')
axes[0].set_title(f'Isolation Forest\nThời gian: {time_if:.2f}s | Phát hiện: {gian_lan_if} giao dịch', fontweight='bold')
axes[0].set_ylabel('Số tiền giao dịch')
axes[0].legend()

# Khung 2: LOF (Màu cam)
axes[1].scatter(df.index[df['Nhan_LOF'] == 1], df['Amount'][df['Nhan_LOF'] == 1], color='blue', alpha=0.3, s=10, label='Bình thường')
axes[1].scatter(df.index[df['Nhan_LOF'] == -1], df['Amount'][df['Nhan_LOF'] == -1], color='orange', marker='X', s=50, label='Gian lận (Dị biệt)')
axes[1].set_title(f'Local Outlier Factor (LOF)\nThời gian: {time_lof:.2f}s | Phát hiện: {gian_lan_lof} giao dịch', fontweight='bold')
axes[1].legend()

# Căn chỉnh và lưu ảnh
plt.tight_layout()
plt.savefig('Bieu_Do_So_Sanh.png')
print("Xong! Đã xuất thành công ảnh 'Bieu_Do_So_Sanh.png'.")
