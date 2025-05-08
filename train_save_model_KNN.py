import numpy as np
import os
import pickle
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Define the path of the dataset
dataset_path = "Dataset/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"

# Define the classes
classes = ["Cyst", "Normal", "Stone", "Tumor"]

# Define the image size
image_size = (200, 200)

# Define the number of neighbors
n_neighbors = 5

# Define the list of features and labels
features = []
labels = []

# Loop through each class and read the images
for class_index, class_name in enumerate(classes):
    class_path = os.path.join(dataset_path, class_name)
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        image = io.imread(image_path)
        image = rgb2gray(image)
        image = resize(image, image_size)
        feature = image.flatten()
        features.append(feature)
        labels.append(class_index)

# Convert the lists into arrays
features = np.array(features)
labels = np.array(labels)

# Split the data into training and testing sets
split_index = int(len(features) * 0.8)
train_features = features[:split_index]
train_labels = labels[:split_index]
test_features = features[split_index:]
test_labels = labels[split_index:]

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(train_features, train_labels)

# Predict the test labels
test_predictions = knn.predict(test_features)

# Compute the accuracy
accuracy = accuracy_score(test_labels, test_predictions)
print("Accuracy:", accuracy)

# Save the model
model_path = "kid_desease_classification_model_KNN.h5"
with open(model_path, "wb") as f:
    pickle.dump(knn, f)
