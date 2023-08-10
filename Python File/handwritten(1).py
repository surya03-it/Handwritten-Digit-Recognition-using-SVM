import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
digits = datasets.load_digits()
X = digits.data
y = digits.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = svm.SVC(gamma=0.001, C=100)
# Train the classifier on the training data
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Classification report:\n", metrics.classification_report(y_test, y_pred))
print("Confusion matrix:\n", metrics.confusion_matrix(y_test, y_pred))
# Select random samples from the test set
sample_indexes = np.random.randint(len(X_test), size=5)
for index in sample_indexes:
    predicted_label = y_pred[index]
    actual_label = y_test[index]
# Reshape the feature vector back into a 2D array (8x8) for visualization
    image = X_test[index].reshape((8, 8))
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(f"Predicted label: {predicted_label}, Actual label: {actual_label}")
    plt.show()
