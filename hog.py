import numpy as np
from skimage import data, color, exposure
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Load an example image dataset
images = data.digits().images
labels = data.digits().target

# Convert the images to grayscale and extract HOG features
features = []
for image in images:
    image = color.rgb2gray(image)
    hog_features = hog(image, orientations=9, pixels_per_cell=(14, 14),
                       cells_per_block=(1, 1), block_norm='L2-Hys',
                       transform_sqrt=True, feature_vector=False)
    features.append(hog_features.ravel())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

# Train an SVM classifier on the HOG features using grid search for hyperparameter tuning
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'loss': ['hinge', 'squared_hinge']}
clf = GridSearchCV(svm.LinearSVC(), param_grid, cv=5)
clf.fit(X_train, y_train)

# Evaluate the best classifier on the testing set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Best parameters:", clf.best_params_)
print("Accuracy:", accuracy)
