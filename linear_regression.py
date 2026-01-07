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

# Target variable
scaler_y = StandardScaler()
y = scaler_y.fit_transform(df['Charges'].values.reshape(-1,1)).flatten().astype(np.float64)

print("X sample:\n", X[:5])
print("y sample:\n", y[:5])

# Batch Gradient Descent
def linear_regression_batch_gradient_descent(X, y, epochs=500, learning_rate=1e-3):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features, dtype=np.float64)
    bias = 0.0

    for epoch in range(epochs):
        # Forward pass
        y_pred = np.dot(X, weights) + bias
        print('y pred', y_pred[1])
        error = y_pred - y

        # Compute gradients
        gradient_w = (2 / n_samples) * np.dot(X.T, error)
        gradient_b = (2 / n_samples) * np.sum(error)


        # Update weights and bias
        weights -= learning_rate * gradient_w
        bias -= learning_rate * gradient_b

        if epoch % 50 == 0:
            loss = np.mean(error ** 2)
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    print("Final weights:", weights)
    print("Final bias:", bias)
    return weights, bias



# --- Stochastic Gradient Descent ---
def linear_regression_sgd(X, y, epochs=100, learning_rate=1e-3):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features, dtype=np.float64)
    bias = 0.0

    for epoch in range(epochs):
        # Shuffle data each epoch
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        for i in indices:
            y_pred = np.dot(weights, X[i]) + bias
            error = y_pred - y[i]

            # Update weights and bias
            weights -= learning_rate * error * X[i]
            bias -= learning_rate * error

        # Compute loss every 10 epochs
        if epoch % 10 == 0:
            y_preds = np.dot(X, weights) + bias
            loss = np.mean((y_preds - y) ** 2)
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    print("Final weights:", weights)
    print("Final bias:", bias)




# Mini batch Gradient Descent

def linear_regression_mini_batch(X, y, epochs=100, learning_rate=1e-3, batch_size = 64, option = '1'):
    alpha = 0.01
    beta = 0.01
    n_samples, n_features = X.shape
    weights = np.zeros(n_features, dtype=np.float64)
    bias = 0.0
    for _ in range(epochs):
        permutation = np.random.permutation(len(X))
        X_shuffled = X[permutation]
        Y_shuffled = y[permutation]
        for i in range(0, len(X), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]
            m = len(X_batch)

                # Forward pass
            y_pred = np.dot(X_batch, weights) + bias
            print('y pred', y_pred[1])
            error = y_pred - Y_batch

        
            # Compute gradients

            #option 1: no regularization
            if option == '1':
                gradient_w = (2 / m) * np.dot(X_batch.T, error)
            #option 2: L1 regularization
            elif option == '2': 
                gradient_w = (2 / m) * np.dot(X_batch.T, error) + beta * np.sign(weights)
            #option 3: L2 regularization
            elif option == '3':
                gradient_w = (2 / m) * np.dot(X_batch.T, error) + 2 * alpha * weights
            #option 4: elastic net (l1 and l2)
            else:
                gradient_w = (2 / m) * np.dot(X_batch.T, error) + 2 * alpha * weights + beta * np.sign(weights)
            
            gradient_b = (2 / m) * np.sum(error)

            # Update weights and bias
            weights -= learning_rate * gradient_w
            bias -= learning_rate * gradient_b
    
    print("mini_batch gradient descent")
    print("Final weights:", weights)
    print("Final bias:", bias)




model_selection = input("Which Model Would you like to run: Option 1: Batch Gradient Descent, Option 2: Stochastic Gradient Descent, Option 3: Mini-batch Gradient Descent")

if model_selection == '1':
    linear_regression_batch_gradient_descent(X, y)
if model_selection == '2':
    linear_regression_sgd(X, y)
if model_selection == '3':
    selection = input("Regularization: Option 1: None, Option 2: L1, Option 3: L2, Else: Elastic Net")
    linear_regression_mini_batch(X, y, epochs=100, learning_rate=1e-3, batch_size = 64, option = selection)



