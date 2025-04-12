import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import Birch, OPTICS
from sklearn.metrics import silhouette_score, adjusted_rand_score
from pyclustering.cluster.clarans import clarans
from time import time

# ======== Đọc và xử lý dữ liệu ========
df = pd.read_csv('milktest.csv')

if 'Label' in df.columns:
    labels_true = df['Label']
    df = df.drop(columns=['Label'])
else:
    labels_true = None

# Mã hóa các cột chứa chuỗi
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])

# Chuẩn hóa
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ======== CLARANS ========
from pyclustering.cluster.clarans import clarans
from sklearn.metrics import silhouette_score
import numpy as np
from time import time

# Chạy thuật toán CLARANS với các giá trị k khác nhau
clarans_scores = []
clarans_labels_all = []
clarans_times = []

print("🔷 Đang chạy CLARANS...")

for k in range(2, 10):  # Thử các giá trị k từ 2 đến 9
    start = time()
    
    # Tạo mô hình CLARANS: dữ liệu dạng list, số cụm k, numlocal=1, maxneighbor=5
    model = clarans(X_scaled.tolist(), k, 1, 5)
    model.process()  # Chạy thuật toán

    duration = time() - start
    clarans_times.append(duration)

    # Lấy kết quả phân cụm (danh sách các cụm)
    clusters = model.get_clusters()
    
    # Tạo nhãn tương ứng cho từng điểm dữ liệu
    labels = np.zeros(len(X_scaled), dtype=int)
    for i, cluster in enumerate(clusters):
        for index in cluster:
            labels[index] = i

    clarans_labels_all.append(labels)
    
    # Tính điểm silhouette để đánh giá chất lượng phân cụm
    score = silhouette_score(X_scaled, labels)
    clarans_scores.append(score)

    print(f"✅ k={k}: Silhouette = {score:.3f}, Thời gian = {duration:.2f}s")

# Tìm k tốt nhất dựa vào silhouette score
best_k = np.argmax(clarans_scores) + 2
clarans_labels = clarans_labels_all[best_k - 2]

print(f"\n🔹 CLARANS tốt nhất: k = {best_k}, Silhouette = {max(clarans_scores):.3f}")

# ======== BIRCH ========
print("\n🔷 Đang chạy BIRCH...")
birch_scores = []
birch_labels_all = []

for k in range(2, 10):
    model = Birch(n_clusters=k)
    labels = model.fit_predict(X_scaled)
    birch_labels_all.append(labels)
    score = silhouette_score(X_scaled, labels)
    birch_scores.append(score)
    print(f"✅ BIRCH k={k}: Silhouette = {score:.3f}")

best_k_birch = np.argmax(birch_scores) + 2
birch_labels = birch_labels_all[best_k_birch - 2]

# ======== OPTICS ========
print("\n🔷 Đang chạy OPTICS...")
optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05)
optics_labels = optics.fit_predict(X_scaled)

if len(set(optics_labels)) > 1:
    optics_score = silhouette_score(X_scaled, optics_labels)
else:
    optics_score = -1
print(f"✅ OPTICS: Silhouette = {optics_score:.3f}")

# ======== Đánh giá ========
print("\n📊 Đánh giá tổng hợp:")
print(f"CLARANS: k = {best_k}, Silhouette = {max(clarans_scores):.3f}")
print(f"BIRCH:   k = {best_k_birch}, Silhouette = {max(birch_scores):.3f}")
print(f"OPTICS:  Silhouette = {optics_score:.3f}")

if labels_true is not None:
    print("\n🎯 ARI so với nhãn gốc:")
    print(f"CLARANS ARI: {adjusted_rand_score(labels_true, clarans_labels):.3f}")
    print(f"BIRCH ARI:   {adjusted_rand_score(labels_true, birch_labels):.3f}")
    print(f"OPTICS ARI:  {adjusted_rand_score(labels_true, optics_labels):.3f}")

# ======== Trực quan hóa ========
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clarans_labels, cmap='tab10')
plt.title(f'CLARANS (k={best_k})')

plt.subplot(1, 3, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=birch_labels, cmap='tab10')
plt.title(f'BIRCH (k={best_k_birch})')

plt.subplot(1, 3, 3)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=optics_labels, cmap='tab10')
plt.title('OPTICS')

plt.tight_layout()
plt.show()
