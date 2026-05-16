import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

print("="*60)
print("PHẦN 1: KẾT QUẢ CÁC BÀI TOÁN PYTHON CƠ BẢN")
print("="*60)

def tinh_giai_thua(n):
    if n < 0: return "Không có giai thừa cho số âm"
    return 1 if n in (0, 1) else n * tinh_giai_thua(n - 1)

def tinh_trung_binh(day_so):
    return sum(day_so) / len(day_so) if day_so else 0

def tinh_loi_nhuan(von_ban_dau, lai_suat_thang, so_thang=12):
    tong_tien = von_ban_dau * ((1 + lai_suat_thang) ** so_thang)
    return tong_tien - von_ban_dau

print(f"- Giai thừa của 5 là: {tinh_giai_thua(5)}")
print(f"- Trung bình dãy số [10, 20, 30, 40, 50] là: {tinh_trung_binh([10, 20, 30, 40, 50])}")
print(f"- Lợi nhuận sau 12 tháng (Vốn 100tr, lãi 0.5%/tháng): {tinh_loi_nhuan(100000000, 0.005):,.0f} VND\n")


print("="*60)
print("PHẦN 2: TIẾN HÀNH CHẠY MÔ HÌNH MACHINE LEARNING PHÁT HIỆN GIAN LẬN")
print("="*60)

df = pd.read_csv("financial_anomaly_data.csv")
print(f"1. Tải dữ liệu thành công! Kích thước file dữ liệu: {df.shape[0]:,} dòng và {df.shape[1]} cột.")

X = df[['Amount']].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("2. Tiền xử lý và chuẩn hóa dữ liệu hoàn tất.")

model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model.fit(X_scaled)

X['Anomaly'] = model.predict(X_scaled)
X['Label'] = X['Anomaly'].map({1: 'Bình thường', -1: 'Bất thường (Gian lận)'})

print("\n3. THỐNG KÊ KẾT QUẢ PHÁT HIỆN GIAN LẬN:")
print("-" * 45)
print(X['Label'].value_counts())
print("-" * 45)

print("\n4. Đang tạo biểu đồ trực quan hóa dữ liệu...")
plt.figure(figsize=(10, 6))

normal_data = X[X['Anomaly'] == 1]
anomaly_data = X[X['Anomaly'] == -1]

# Dùng lệnh scatter thuần túy nhất, không thể lỗi được nữa
plt.scatter(normal_data.index, normal_data['Amount'], color='blue', label='Bình thường', alpha=0.5)
plt.scatter(anomaly_data.index, anomaly_data['Amount'], color='red', label='Gian lận', s=50, marker='X')

plt.title('BIEU DO PHAT HIEN GIAO DICH GIAN LAN')
plt.xlabel('Thu tu giao dich')
plt.ylabel('So tien (Amount)')
plt.legend()
print("5. Vẽ biểu đồ thành công! Vui lòng xem bảng biểu đồ vừa hiện lên.")
plt.show()