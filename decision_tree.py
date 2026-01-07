import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

y = df['Charges']

print("X sample:\n", X[:5])
print("y sample:\n", y[:5])


class Branch:
    def __init__(self, left=None, right=None, threshold=None, feature=None, value=None):
        self.left = left
        self.right = right
        self.threshold = threshold
        self.feature = feature
        self.value = value


n_samples, n_features = X.shape

max_depth = None
min_samples_split = 2
min_samples_leaf = 5
min_impurity_decrease = 0.0
max_leaf_nodes = None


def variance_reduction_decision_tree():
    current_depth, current_samples_split, current_samples_leaf, current_leaf_nodes = 0
    while current_depth <= max_depth or current_samples_split >= min_samples_split or current_samples_leaf >= min_samples_leaf or current_leaf_nodes <= max_leaf_nodes:
        best_variance_reduction = None
        




def best_split(X, y):
    for i in range(0, n_features):
            feature = X[:, i]
            idx = np.argsort(feature)
            feature_sorted = feature[idx]
            y_sorted = y[idx]
    for j in range(0, len(feature_sorted) - 1):
                if feature_sorted[j] == feature_sorted[j+1]:
                    continue
                else:
                    left_side_X = feature_sorted[:j+1]
                    left_side_y = y_sorted[:j+1]
                    right_side_X = feature_sorted[j+1:]
                    right_side_y = y_sorted[j+1:]


                



