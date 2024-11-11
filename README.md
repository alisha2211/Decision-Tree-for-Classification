# Decision-Tree-for-Classification

from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

# Create a dataset
X = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [3, 3], [4, 4]])
y = np.array([0, 0, 0, 1, 1, 1])  # Labels for two classes

# Initialize and train the decision tree classifier
tree = DecisionTreeClassifier()
tree.fit(X, y)

# Predict class for a new point
x_new = np.array([[2, 3]])
predicted_class = tree.predict(x_new)
print("Predicted class for new point:", predicted_class[0])

# Visualize decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=100)
plt.scatter(x_new[0, 0], x_new[0, 1], c='green', marker='*', s=200, label="New Point")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.title("Decision Tree Classification")
plt.show()
