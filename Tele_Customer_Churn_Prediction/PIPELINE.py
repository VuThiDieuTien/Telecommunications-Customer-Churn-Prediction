import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os
import warnings

# Tắt cảnh báo để output sạch đẹp
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CẤU HÌNH & THIẾT LẬP
# ==============================================================================
# Đường dẫn file gốc (Bạn upload file này cùng thư mục với script)
INPUT_FILE = 'telecom_customer_churn.csv' 

# Thư mục xuất dữ liệu cho Dashboard
OUTPUT_DIR = 'data/dashboard'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("🚀 BẮT ĐẦU PIPELINE XỬ LÝ DỮ LIỆU...")

# ==============================================================================
# 2. TẢI & LÀM SẠCH DỮ LIỆU (DATA CLEANING)
# ==============================================================================
try:
    df_raw = pd.read_csv(INPUT_FILE)
    print(f"✅ Đã tải dữ liệu gốc: {df_raw.shape}")
except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file '{INPUT_FILE}'. Vui lòng upload file này.")
    exit()

# Giữ lại bản copy để làm Master Data cho Dashboard (Dữ liệu Human-Readable)
df_master = df_raw.copy()

# Xử lý Target (Customer Status -> 0/1)
# Chỉ lấy 'Churned' và 'Stayed', bỏ 'Joined' (nếu muốn dự báo hành vi rời bỏ chính xác)
df_model = df_raw[df_raw['Customer Status'].isin(['Churned', 'Stayed'])].copy()
df_model['Churn_Label'] = df_model['Customer Status'].apply(lambda x: 1 if x == 'Churned' else 0)

# Xử lý dữ liệu thiếu (Missing Values) cơ bản
# Điền 0 cho các cột số, Mode cho các cột phân loại
num_cols = df_model.select_dtypes(include=[np.number]).columns
cat_cols = df_model.select_dtypes(include=['object']).columns

for col in num_cols:
    df_model[col] = df_model[col].fillna(0)
    
for col in cat_cols:
    df_model[col] = df_model[col].fillna(df_model[col].mode()[0])

print("✅ Đã xử lý dữ liệu thiếu & Target Label.")

# ==============================================================================
# 3. PHÂN CỤM KHÁCH HÀNG (CLUSTERING - KMEANS)
# ==============================================================================
print("⏳ Đang thực hiện Phân cụm (Clustering)...")

# Chọn features cho clustering (Dựa trên notebook Clustering của bạn)
# Thông thường dùng: Tenure, Monthly Charge, Referrals, Dependents
cluster_features = ['Tenure in Months', 'Monthly Charge', 'Number of Referrals', 'Number of Dependents']

# Kiểm tra xem các cột có tồn tại không
valid_cluster_feats = [c for c in cluster_features if c in df_model.columns]

if valid_cluster_feats:
    # Chuẩn hóa dữ liệu cho Clustering
    scaler_cluster = MinMaxScaler()
    X_cluster = scaler_cluster.fit_transform(df_model[valid_cluster_feats])
    
    # Huấn luyện KMeans (K=4 theo đề xuất Dashboard)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_cluster)
    
    # Gán nhãn cluster vào dữ liệu Master
    # Lưu ý: Cần map lại index cho đúng vì df_model có thể ít dòng hơn df_raw nếu lọc Joined
    df_model['Cluster_Label'] = clusters
    
    # Lưu model clustering
    joblib.dump(kmeans, f'{OUTPUT_DIR}/kmeans_model.pkl')
    joblib.dump(scaler_cluster, f'{OUTPUT_DIR}/cluster_scaler.pkl')
    print("✅ Đã phân cụm xong (4 nhóm).")
else:
    print("⚠️ Thiếu cột dữ liệu cho Clustering. Bỏ qua bước này.")
    df_model['Cluster_Label'] = -1

# ==============================================================================
# 4. MÔ HÌNH DỰ BÁO (MODELING - XGBOOST)
# ==============================================================================
print("⏳ Đang huấn luyện mô hình XGBoost...")

# Chuẩn bị dữ liệu huấn luyện (Preprocessing)
# Bỏ các cột không dùng train (ID, thông tin hành chính, Churn Category - vì là target)
drop_cols = ['Customer ID', 'City', 'Zip Code', 'Latitude', 'Longitude', 
             'Churn Category', 'Churn Reason', 'Customer Status', 'Churn_Label', 'Cluster_Label']
# Xử lý drop an toàn
cols_to_drop = [c for c in drop_cols if c in df_model.columns]

X = df_model.drop(columns=cols_to_drop)
y = df_model['Churn_Label']

# One-Hot Encoding cho biến phân loại
X_encoded = pd.get_dummies(X, drop_first=True)

# Chia Train/Test
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

# Huấn luyện XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)
xgb_model.fit(X_train, y_train)

# Đánh giá nhanh
y_pred = xgb_model.predict(X_test)
auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
print(f"📊 Kết quả Model trên tập Test:")
print(f"   - AUC-ROC: {auc:.4f}")
print(f"   - Accuracy: {(y_pred == y_test).mean():.4f}")

# Lưu Model
joblib.dump(xgb_model, f'{OUTPUT_DIR}/churn_model_xgb.pkl')
# Lưu danh sách cột sau khi one-hot (để dùng khi dự báo data mới)
joblib.dump(X_encoded.columns.tolist(), f'{OUTPUT_DIR}/model_features.pkl')

print("✅ Đã huấn luyện và lưu Model XGBoost.")

# ==============================================================================
# 5. XUẤT DỮ LIỆU CHO DASHBOARD (EXPORT MASTER DATA)
# ==============================================================================
print("⏳ Đang tạo file Master Data cho Dashboard...")

# Dự báo lại trên TOÀN BỘ dữ liệu để Dashboard hiển thị
# (Dashboard cần hiển thị cả khách hàng cũ và dự báo rủi ro cho họ)
all_probs = xgb_model.predict_proba(X_encoded)[:, 1]
all_preds = xgb_model.predict(X_encoded)

# Ghép kết quả vào DataFrame gốc (df_model - phiên bản đã lọc bỏ Joined)
# Lưu ý: Dashboard cần các cột gốc (chữ) + Cột dự báo (số)
df_export = df_model.copy() # Lấy bản đã có Cluster
df_export['Churn_Probability'] = all_probs
df_export['Prediction'] = all_preds

# Mapping tên cột cho chuẩn Dashboard của bạn (để khớp code Dashboard.py)
# Đổi tên cột về dạng chuẩn (snake_case hoặc CamelCase tùy Dashboard)
rename_map = {
    'Tenure in Months': 'tenure_in_months',
    'Monthly Charge': 'monthly_charge',
    'Number of Referrals': 'number_of_referrals',
    'Number of Dependents': 'number_of_dependents',
    'Contract': 'contract', # Để code Dashboard nhận diện
    'Payment Method': 'payment_method',
    'Internet Type': 'internet_type',
    'Cluster_Label': 'Cluster_Label',
    'Customer Status': 'customer_status'
}
df_export.rename(columns=rename_map, inplace=True)

# Tạo cột Persona Name (cho đẹp Dashboard)
persona_map = {
    0: 'Persona 1: Rủi ro cao', 
    1: 'Persona 2: Tiết kiệm', 
    2: 'Persona 3: Ổn định', 
    3: 'Persona 4: Trung thành'
}
# Lưu ý: Cần map lại persona theo logic thực tế (ở đây map tạm, Dashboard sẽ tự tính lại)
# Trong Pipeline này ta chỉ cần xuất cột Cluster_Label chuẩn là được.

# Lưu file CSV Master
output_csv = f'{OUTPUT_DIR}/dashboard_master_data.csv'
df_export.to_csv(output_csv, index=False)

# Lưu thêm metrics để hiển thị
metrics = {
    'accuracy': float((y_pred == y_test).mean()),
    'auc': float(auc),
    'precision': float(classification_report(y_test, y_pred, output_dict=True)['1']['precision']),
    'recall': float(classification_report(y_test, y_pred, output_dict=True)['1']['recall'])
}
import json
with open(f'{OUTPUT_DIR}/model_metrics.json', 'w') as f:
    json.dump(metrics, f)

# Feature Importance
importance_df = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)
importance_df.to_csv(f'{OUTPUT_DIR}/feature_importance.csv', index=False)
