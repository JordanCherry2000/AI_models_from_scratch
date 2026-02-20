import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load data
column_names = ['Age','Sex', 'BMI', 'Children', 'Smoker', 'Region', 'Charges']
df = pd.read_csv("Medical-Insurance.csv", header=None, names=column_names)

# Convert all numeric columns, safely coerce errors
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
df['Children'] = pd.to_numeric(df['Children'], errors='coerce')
df['Charges'] = pd.to_numeric(df['Charges'], errors='coerce')

# Drop rows with any NaN values
df = df.dropna()

# Numeric features
numeric = df[['Age','BMI','Children']]
scaler_X = StandardScaler()
numeric_scaled = scaler_X.fit_transform(numeric).astype(np.float64)

# Categorical features
sex_encoded = pd.get_dummies(df['Sex'], drop_first=True).astype(np.float64)
smoker_encoded = pd.get_dummies(df['Smoker'], drop_first=True).astype(np.float64)
region_encoded = pd.get_dummies(df['Region'], drop_first=True).astype(np.float64)

# Combine all features
X = np.hstack([numeric_scaled, sex_encoded.to_numpy(), smoker_encoded.to_numpy(), region_encoded.to_numpy()]).astype(np.float64)

y = df['Charges'].to_numpy(dtype=np.float64)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


class Node:
    def __init__(self, left, right, threshold, feature_index, leaf_value):
        self.left = left
        self.right = right
        self.threshold = threshold
        self.feature_index = feature_index
        self.leaf_value = leaf_value


max_depth = 6
min_impurity_decrease = 0.0
min_leaf_samples = 2
min_sample_split = 10
max_leaf_nodes = 3


def mse_of_y(y_vals):
    if y_vals.size == 0:
        return 0.0
    mu = y_vals.mean()
    return np.mean((y_vals - mu) **2)


def get_best_split(X, y, indices):
    n_samples, n_features = X.shape
    n_node = indices.size
    features_sqrt = int(np.sqrt(n_features))
    selected_features = np.random.choice(n_features, size=features_sqrt, replace=False)
    best_gain = 0.0
    best_split = None
    best_threshold = None
    best_feature_index = None
    parent_mse = mse_of_y(y[indices])
    if n_node < min_sample_split:
        return None, 0, None
    for i in selected_features:
        feature_values = X[indices, i]
        order = np.argsort(feature_values)
        sorted_indices = indices[order]
        vals = X[sorted_indices, i]
        y_vals = y[sorted_indices]
        for j in range(0, len(sorted_indices) - 1):
            k = j + 1
            if vals[j] == vals[k]:
                continue
            y_left = y_vals[:k]
            y_right = y_vals[k:]
            threshold = (vals[j] + vals[k]) / 2
            left_split_mse = mse_of_y(y_left)
            right_split_mse = mse_of_y(y_right)
            weighted_mse = (k/n_node) * left_split_mse + ((n_node - k) /n_node) * right_split_mse
            gain = parent_mse - weighted_mse
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
                best_feature_index = i
    return best_feature_index, best_gain, best_threshold




def build_tree(X, y, indices, depth):
    if depth >= max_depth or len(indices) < min_leaf_samples:
        leaf_value = y[indices].mean()
        return Node(None, None, None, None, leaf_value)
    feature, gain, threshold = get_best_split(X, y, indices)
    
    if feature is None or gain < min_impurity_decrease:
        leaf_value = y[indices].mean()
        return Node(None, None, None, None, leaf_value)
    

    left_indices = indices[X[indices, feature] <= threshold]
    right_indices = indices[X[indices, feature] > threshold]

    left_child = build_tree(X, y, left_indices, depth + 1)
    right_child = build_tree(X, y, right_indices, depth + 1)



    return Node(left_child, right_child, threshold, feature, None)


def predict_tree(X, tree):
    predictions = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        node = tree
        while node.leaf_value is None:
            if X[i, node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        predictions[i] = node.leaf_value
    return predictions



def build_rf(X, y, n_trees=10):
    n_samples, n_features = X_train.shape
    indices = np.arange(n_samples)
    rf = []
    for i in range(n_trees):
        new_indices = np.random.choice(indices, size=len(indices), replace=True)
        rf.append(build_tree(X, y, new_indices, depth=0))
    return rf 

def predict_rf(X_test, rf):
    n_samples = X_test.shape[0]
    predictions = np.zeros(n_samples) 

    for tree in rf:
        predictions += predict_tree(X_test, tree) 

    predictions /= len(rf)
    return predictions

rf = build_rf(X, y, n_trees=10)
predictions = predict_rf(X_test, rf)

mse = np.mean((predictions - y_test) **2)
print("RF MSE: ",mse)