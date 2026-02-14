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



def naive_bayes_training(X_train, y_train):
    n_samples, n_features = X_train.shape
    classes = np.unique(y_train)
    eps = 1e-9
    grouped = {}
    mean = np.zeros((len(classes), n_features), dtype=np.float64)
    std = np.zeros((len(classes), n_features), dtype=np.float64)
    prior = np.zeros(len(classes))
    for c in range(len(classes)):
        grouped[c] = X_train[y_train == classes[c]]
        prior[c] = len(grouped[c])/n_samples
        for i in range(n_features):
            mean[c][i] = np.mean(grouped[c][:,i])
            std[c][i] = np.std(grouped[c][:,i]) + eps
    
    
    return mean, std, prior, grouped


def gaussian_likelihood(x, mean, std):
    exponent = -0.5 * ((x - mean) / std) ** 2
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(exponent)


def naive_bayes_test(X_test, mean, std, prior):
    n_samples, n_features = X_test.shape
    n_classes = len(prior)
    y_pred = np.empty(n_samples, dtype=object) 



    for idx, x in enumerate(X_test):
        posteriors = np.zeros(n_classes)

        for c in range(n_classes):
            likelihood = gaussian_likelihood(x, mean[c], std[c])
            likelihood_prod = np.prod(likelihood)

            posteriors[c] = likelihood_prod * prior[c]

        y_pred[idx] = classes[np.argmax(posteriors)]
    return y_pred



mean, std, prior, grouped = naive_bayes_training(X_train, y_train)

y_pred = naive_bayes_test(X_test, mean, std, prior)

print(np.mean(y_pred == y_test))
    
