import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

column_names = ['Age','Sex', 'BMI', 'Children', 'Smoker', 'Region', 'Charges']
df = pd.read_csv("Medical-Insurance.csv", header=None, names=column_names)


df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
df['Children'] = pd.to_numeric(df['Children'], errors='coerce')
df['Charges'] = pd.to_numeric(df['Charges'], errors='coerce')


df = df.dropna()

scaler_X = StandardScaler()
numeric_var = df[['Age', 'BMI', 'Children']]

numeric_scaled = scaler_X.fit_transform(numeric_var).astype(np.float64)

sex_encoded = pd.get_dummies(df['Sex'], drop_first=True).astype(np.float64)
smoker_encoded = pd.get_dummies(df['Smoker'], drop_first=True).astype(np.float64)
region_encoded = pd.get_dummies(df['Region'], drop_first=True).astype(np.float64)

X = np.hstack([numeric_scaled, sex_encoded.to_numpy(), smoker_encoded.to_numpy(), region_encoded.to_numpy()]).astype(np.float64)


scaler_y = StandardScaler()
y = scaler_y.fit_transform(df['Charges'].values.reshape(-1,1)).flatten().astype(np.float64)


def batch_gradient_descent (X, y, learning_rate, epochs):
    n_samples, n_features = X.shape
    weights, bias = gradient_descent(X, y, learning_rate, epochs)


def mini_batch_gradient_descent (X, y, learning_rate, epochs, batch_size):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features, dtype=np.float64)
    bias = 0.0
    for _ in range(epochs):
        permutation = np.random.permutation(len(X))
        X_shuffled = X[permutation]
        Y_shuffled = y[permutation]
        y_pred = np.dot(X_shuffled[:batch_size], weights) + bias
        error = y_pred - Y_shuffled[:batch_size]
        gradient_w = (2/n_samples) * np.dot(X.T, error)
        gradient_b = (2/n_samples) * np.sum(error)

        weights -= learning_rate * gradient_w
        bias -= learning_rate * gradient_b

        return weights, bias
        print(weights, bias)
    

def gradient_descent (X, y, learning_rate, epochs):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features, dtype=np.float64)
    bias = 0.0
    for _ in range(epochs):
        y_pred = np.dot(X, weights) + bias
        error = y_pred - y
        gradient_w = (2/n_samples) * np.dot(X.T, error)
        gradient_b = (2/n_samples) * np.sum(error)

        weights -= learning_rate * gradient_w
        bias -= learning_rate * gradient_b

        return weights, bias

    

batch_gradient_descent(X, y, 0.05, 200)
mini_batch_gradient_descent(X, y, 0.05, 200, 64)