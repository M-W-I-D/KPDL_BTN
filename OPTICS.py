import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Hàm tính khoảng cách Euclidean
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# Hàm tìm các điểm lân cận của một điểm với eps
def get_neighbors(data, point_idx, eps):
    neighbors = []
    for i in range(len(data)):
        if i != point_idx and euclidean_distance(data[point_idx], data[i]) <= eps:
            neighbors.append(i)
    return neighbors

# Hàm thực hiện thuật toán OPTICS
def optics(data, min_samples, eps):
    n = len(data)
    reachability = np.full(n, np.inf)  # Khởi tạo giá trị reachability cho mỗi điểm
    processed = np.zeros(n, dtype=bool)  # Mảng để kiểm tra các điểm đã được xử lý
    order = []  # Danh sách lưu trữ thứ tự điểm được xử lý
    clusters = []  # Lưu trữ các cụm
    seeds = {}  # Lưu trữ các điểm chưa được xử lý

    for i in range(n):
        if processed[i]:
            continue  # Bỏ qua các điểm đã xử lý

        neighbors = get_neighbors(data, i, eps)
        processed[i] = True
        order.append(i)

        if len(neighbors) < min_samples:
            continue  # Nếu số điểm lân cận ít hơn min_samples, không tạo cụm

        for idx in neighbors:
            reachability[idx] = euclidean_distance(data[i], data[idx])
            seeds[idx] = reachability[idx]

        while seeds:
            next_idx = min(seeds, key=seeds.get)  # Lấy điểm tiếp theo có reachability nhỏ nhất
            reachability[next_idx] = seeds[next_idx]
            processed[next_idx] = True
            order.append(next_idx)
            del seeds[next_idx]

            # Tìm các điểm lân cận của điểm tiếp theo
            new_neighbors = get_neighbors(data, next_idx, eps)
            if len(new_neighbors) >= min_samples:
                for j in new_neighbors:
                    if not processed[j]:
                        new_reach = max(reachability[next_idx], euclidean_distance(data[next_idx], data[j]))
                        if reachability[j] == np.inf or new_reach < reachability[j]:
                            reachability[j] = new_reach
                            seeds[j] = new_reach

    # Xác định số lượng cụm từ reachability
    cluster_labels = [-1] * n  # -1 nghĩa là không có cụm, các số dương là ID của các cụm
    cluster_id = 0
    for i in range(n):
        if cluster_labels[i] == -1:  # Chưa được phân cụm
            # Tạo một cụm mới
            cluster_labels[i] = cluster_id
            cluster_id += 1
    return order, reachability, cluster_labels

# Đọc dữ liệu
df = pd.read_csv("C:/KPDL/BTN/Students_Grading_Dataset.csv")

# Chọn các cột liên quan đến điểm số và các yếu tố ảnh hưởng
features = ['Midterm_Score', 'Final_Score', 'Study_Hours_per_Week', 'Participation_Score', 'Sleep_Hours_per_Night']

# Kiểm tra xem các cột có trong dữ liệu hay không
missing_columns = [col for col in features if col not in df.columns]
if missing_columns:
    print(f"Các cột không có trong dữ liệu: {missing_columns}")

# Nếu có cột thiếu, sẽ chỉ sử dụng những cột có sẵn
df_selected = df[[col for col in features if col in df.columns]]

# Chuẩn hóa dữ liệu (scaling)
df_scaled = (df_selected - df_selected.mean()) / df_selected.std()

# Chạy thuật toán OPTICS với tham số đã điều chỉnh để giảm bớt số lượng cụm
eps = 4  # Tăng eps để các điểm có thể được phân nhóm chung
min_samples = 15  # Tăng min_samples để yêu cầu nhiều điểm hơn để tạo thành một cụm

order, reachability, cluster_labels = optics(df_scaled.values, min_samples, eps)

# Thêm cột "Total_Score" vào để dễ dàng phân tích kết quả
df['Total_Score'] = df['Midterm_Score'] + df['Final_Score'] + df['Participation_Score'] + df['Projects_Score']

# Phân tích học sinh có điểm cao nhất (ví dụ: Tổng điểm > 80)
high_score_students = df[df['Total_Score'] > 80]

# Vẽ biểu đồ phân cụm với các học sinh có điểm cao
plt.figure(figsize=(10, 6))
plt.scatter(df['Study_Hours_per_Week'], df['Total_Score'], c=cluster_labels, cmap='viridis', marker='o')
plt.xlabel('Study Hours per Week')
plt.ylabel('Total Score')
plt.title('Phân cụm học sinh và ảnh hưởng của thời gian học đến điểm số')
plt.colorbar(label='Cluster ID')
plt.show()

# Vẽ biểu đồ phân cụm với học sinh có điểm cao (Total_Score > 80)
plt.figure(figsize=(10, 6))
plt.scatter(high_score_students['Study_Hours_per_Week'], high_score_students['Total_Score'], c=cluster_labels, cmap='viridis', marker='o')
plt.xlabel('Study Hours per Week')
plt.ylabel('Total Score')
plt.title('Phân cụm học sinh có điểm cao')
plt.colorbar(label='Cluster ID')
plt.show()

# Xem kết quả phân cụm
print(f"Số lượng cụm: {len(set(cluster_labels)) - 1}")  # Trừ 1 để không tính cụm -1 (các điểm không thuộc cụm nào)
print(f"Số học sinh có điểm cao nhất: {high_score_students.shape[0]}")
print(high_score_students[['Student_ID', 'Total_Score']])
