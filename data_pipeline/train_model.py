import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load data
X = np.load("dataset.npy")
y = np.load("labels.npy")

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))