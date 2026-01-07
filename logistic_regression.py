import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("framingham.csv")

print(df.dtypes)

df['cigsPerDay'] = pd.to_numeric(df['cigsPerDay'], errors='coerce')
df['totChol'] = pd.to_numeric(df['totChol'], errors='coerce')
df['sysBP'] = pd.to_numeric(df['sysBP'], errors='coerce')
df['diaBP'] = pd.to_numeric(df['diaBP'], errors='coerce')
df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
df['heartRate'] = pd.to_numeric(df['heartRate'], errors='coerce')
df['glucose'] = pd.to_numeric(df['glucose'], errors='coerce')


print(df.shape)
df = df.dropna()
print(df.shape)

numeric = df[['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']]
scaler_X = StandardScaler()
numeric_scaled = scaler_X.fit_transform(numeric).astype(np.float64)

print(numeric_scaled)

male_encoded = pd.get_dummies(df['male'], drop_first=True).astype(np.float64)
education_encoded = pd.get_dummies(df['education'], drop_first=True).astype(np.float64)
currentSmoker_encoded = pd.get_dummies(df['currentSmoker'], drop_first=True).astype(np.float64)
BPMeds_encoded = pd.get_dummies(df['BPMeds'], drop_first=True).astype(np.float64)
prevalentStroke_encoded = pd.get_dummies(df['prevalentStroke'], drop_first=True).astype(np.float64)
prevalentHyp_encoded = pd.get_dummies(df['prevalentHyp'], drop_first=True).astype(np.float64)


X = np.hstack([numeric_scaled, male_encoded.to_numpy(), education_encoded.to_numpy(), currentSmoker_encoded.to_numpy(), BPMeds_encoded.to_numpy(), prevalentStroke_encoded.to_numpy(), prevalentHyp_encoded.to_numpy()]).astype(np.float64)


y = df['TenYearCHD']

print(X[:5])
print(y[:5])


# Batch Gradient Descent
def logistic_regression_batch_gradient_descent(X, y, epochs=500, learning_rate=1e-3):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features, dtype=np.float64)
    bias = 0.0
    for epoch in range(epochs):
        # Forward pass
        z = np.dot(X, weights) + bias
        y_pred = 1/(1 + np.exp(-z))
        print('y pred', y_pred[1])
        error = y_pred - y

        # Compute gradients
        gradient_w = (1 / n_samples) * np.dot(X.T, error)
        gradient_b = (1 / n_samples) * np.sum(error)

        # Update weights and bias
        weights -= learning_rate * gradient_w
        bias -= learning_rate * gradient_b

        if epoch % 50 == 0:
            loss = -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))
            print(f"Epoch {epoch}, Loss: {loss:.6f}")


    

    print("Final weights:", weights)
    print("Final bias:", bias)
    return weights, bias


logistic_regression_batch_gradient_descent(X, y, epochs = 500, learning_rate = 0.001)