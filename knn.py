import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter


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



def euclidean_algorithm(point_1, point_2):
    return np.sqrt(np.sum((point_1 - point_2) ** 2))

def single_prediction(current_val, X_train, y_train, k=3):
    distances = [euclidean_algorithm(current_val, x) for x in X_train]
    closest_k = np.argsort(distances)[:k]
    nearest_labels = [y_train[i] for i in closest_k]
    final_total = Counter(nearest_labels).most_common(1)[0][0]

    return final_total


def knn(X_test, X_train, y_test, y_train, k=3):
    total_predictions = [single_prediction(x, X_train, y_train, k=3) for x in X_test]
    return total_predictions

predictions = knn(X_test, X_train, y_test, y_train, k=3)

print(np.mean(predictions == y_test))
