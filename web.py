import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
from dotenv import load_dotenv

# --- CÀI ĐẶT GIAO DIỆN TRANG WEB ---
st.set_page_config(page_title="Hệ thống Phát hiện Gian lận", layout="wide")
st.title("🕵️ Hệ Thống Phát Hiện Gian Lận Tài Chính Bằng AI")
st.markdown("**Sinh viên thực hiện:** Hoàng Đặng Mai Linh - Lê Thị Kiều")
st.markdown("---")

# --- XỬ LÝ API KEY ---
load_dotenv()
REAL_API_KEY = os.getenv("SECRET_API_KEY")
if REAL_API_KEY:
    st.success(f"🔑 Trạng thái API: Đã kết nối an toàn (Key: {REAL_API_KEY[:5]}********)")
else:
    st.error("🔑 Cảnh báo: Chưa kết nối API Key!")

# --- CHẠY AI VÀ HIỂN THỊ KẾT QUẢ ---
st.header("1. Phân tích Dữ liệu bằng Isolation Forest")

try:
    # Tải dữ liệu
    df = pd.read_csv("financial_anomaly_data.csv")
    st.write(f"Đã nạp thành công **{df.shape[0]:,}** giao dịch vào hệ thống.")
    
    # Tiền xử lý
    X = df[['Amount']].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Huấn luyện mô hình
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    model.fit(X_scaled)
    
    # Gắn nhãn
    X['Anomaly'] = model.predict(X_scaled)
    
    # Chia 2 cột để hiển thị số liệu cho đẹp
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"✅ Giao dịch Bình thường: **{len(X[X['Anomaly']==1]):,}**")
    with col2:
        st.error(f"🚨 Giao dịch Có Khả Nang Cao Là Gian Lận: **{len(X[X['Anomaly']==-1]):,}**")

    # --- VẼ BIỂU ĐỒ LÊN WEB ---
    st.header("2. Biểu Đồ Trực Quan Hóa")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    normal_data = X[X['Anomaly'] == 1]
    anomaly_data = X[X['Anomaly'] == -1]
    
    ax.scatter(normal_data.index, normal_data['Amount'], color='blue', label='Bình thường', alpha=0.5)
    ax.scatter(anomaly_data.index, anomaly_data['Amount'], color='red', label='Gian lận', s=50, marker='X')
    
    ax.set_title('PHÂN BỐ GIAO DỊCH VÀ CÁC ĐIỂM DỊ BIỆT')
    ax.set_xlabel('Thứ tự giao dịch')
    ax.set_ylabel('Số tiền')
    ax.legend()
    
    # Hiển thị hình ảnh lên web
    st.pyplot(fig)
    st.balloons() # Hiệu ứng bóng bay chúc mừng hoàn thành

except Exception as e:
    st.error(f"Lỗi không tìm thấy file dữ liệu: {e}")