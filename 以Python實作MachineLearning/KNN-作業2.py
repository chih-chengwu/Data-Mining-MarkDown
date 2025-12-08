import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 原始資料
data = np.array([
    [58, 9, 1],
    [30, 6, 0],
    [37, 12, 1],
    [70, 12, 0],
    [40, 5, 0],
    [27, 7, 0],
    [39, 13, 1],
    [52, 6, 1],
    [61, 8, 1],
    [44, 14, 1],
    [62, 17, 0],
    [18, 5, 0],
    [16, 0, 0],
    [18, 12, 0],
    [71, 2, 0],
    [60, 8, 1],
    [46, 9, 1],
    [58, 9, 1],
    [48, 5, 0],
    [46, 6, 0],
    [47, 10, 1],
    [36, 18, 0]
])

# 分割特徵與標籤
X = data[:, :2]   # Age, Income
y = data[:, 2]    # Tour

# 前18筆為訓練資料
X_train = X[:18]
y_train = y[:18]

# 最後4筆為測試資料
X_test = X[18:]
y_test = y[18:]

# 進行標準化（IMPORTANT：train fit，再 transform test）
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# 建立 KNN 模型 (可用預設 k=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_std, y_train)

# 預測
y_pred = knn.predict(X_test_std)

# 計算正確率
acc = accuracy_score(y_test, y_pred)

print("最後 4 筆資料預測 Tour：", y_pred)
print("實際 Tour：", y_test)
print("正確率：", acc)
