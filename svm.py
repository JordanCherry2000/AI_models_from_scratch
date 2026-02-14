import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


classes_to_keep = ["Iris-setosa", "Iris-versicolor"]
df = pd.read_csv("Iris.csv").drop(columns=['Id'])
df = df[df["Species"].isin(classes_to_keep)]

X = df.drop(columns=["Species"]).values
y = df["Species"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()

classes = np.unique(y_train)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = np.where(y_train == classes[0], -1, 1)
y_test = np.where(y_test == classes[0], -1, 1)



def svm(X_train, y_train, n_iters=1000, lambda_param = 0.05, learning_rate = 0.001):
    n_samples, n_features = X_train.shape
    w = np.zeros(n_features)
    b = 0.0
    for i in range(n_iters):
        for idx, x_i in enumerate(X_train):
            condition = y_train[idx] * (np.dot(x_i, w) - b) >= 1
            if condition:
                w -= learning_rate * (2 * lambda_param * w)
            else:
                w -= learning_rate * (2 * lambda_param * w - x_i * y_train[idx])
                b -= learning_rate * y_train[idx]
        print(w, b)
    return w, b



def predict(X_test, w, b):
    approx = np.dot(X_test, w) - b
    return np.sign(approx)


w, b = svm(X_train, y_train, n_iters=1000, lambda_param=0.05, learning_rate=0.001)

predict(X_test, w, b)