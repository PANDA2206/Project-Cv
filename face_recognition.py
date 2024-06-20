import cv2
import numpy as np
import pickle
import os
import random
from scipy import spatial
from sklearn.cluster import KMeans
from classifier import NearestNeighborClassifier


# FaceNet to extract face embeddings.
class FaceNet:

    def __init__(self):
        self.dim_embeddings = 128
        self.facenet = cv2.dnn.readNetFromONNX("resnet50_128.onnx")

    # Predict embedding from a given face image.
    def predict(self, face):
        # Normalize face image using mean subtraction.
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) - (131.0912, 103.8827, 91.4953)

        # Forward pass through deep neural network. The input size should be 224 x 224.

        reshaped = np.moveaxis(face, 2, 0)
        reshaped = np.expand_dims(reshaped, axis=0)
        self.facenet.setInput(reshaped)
        embedding = np.squeeze(self.facenet.forward())
        return embedding / np.linalg.norm(embedding)

        # changing the axis and channel axis first because convieniecy for neural network
        # addiding dimension at 0 th column to add batch size or also 1*4 for facenet
        # Input the facenet data
        #sequezzing the remove single dimensions entries from shape of array getting emedding
        # embedding magnitude doesn't important direction important so normalize it.

    # Get dimensionality of the extracted embeddings.
    def get_embedding_dimensionality(self):
        return self.dim_embeddings


# The FaceRecognizer model enables supervised face identification.
class FaceRecognizer:

    # Prepare FaceRecognizer; specify all parameters for face identification.
    def __init__(self, num_neighbours=11, max_distance=0.8, min_prob=0.5):
        # ToDo: Prepare FaceNet and set all parameters for kNN.
        self.facenet = FaceNet()
        self.num_neighbours = num_neighbours
        self.max_distance = max_distance
        self.min_prob = min_prob
        self.nearest_neighbor_classifier = NearestNeighborClassifier()
        
        # The underlying gallery: class labels and embeddings.
        self.labels = []
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))

        # Load face recognizer from pickle file if available.
        if os.path.exists("recognition_gallery.pkl"):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open("recognition_gallery.pkl", 'wb') as f:
            pickle.dump((self.labels, self.embeddings), f)

    # Load trained model from a pickle file.
    def load(self):
        with open("recognition_gallery.pkl", 'rb') as f:
            (self.labels, self.embeddings) = pickle.load(f)

    # ToDo
    def update(self, face, label):
        embedding = self.facenet.predict(face)
        self.labels.append(label)
        self.embeddings = np.vstack((self.embeddings, embedding))
        # Update NearestNeighborClassifier
        self.nearest_neighbor_classifier.fit(self.embeddings, np.array(self.labels))
        self.save()

    # ToDo
    def predict(self, face):
        embedding = self.facenet.predict(face)

        # Use k-nearest neighbors to find the nearest embeddings in the gallery
        prediction_labels,similarities = self.nearest_neighbor_classifier.predict_labels_and_similarities(np.array([embedding]))

        # # Convert numeric labels back to strings
        # predicted_labels = self.label_encoder.inverse_transform(prediction_labels)

        # Calculate posterior probability and distance for each class
        posterior_probs = {}
        distances = {}

        for label in np.unique(self.labels):
            label_indices = np.where(np.array(self.labels) == label)[0]
            label_distances = similarities[label_indices]
            ki = min(len(label_distances), self.nearest_neighbor_classifier.k)

            # Calculate posterior probability
            posterior_prob = ki / self.nearest_neighbor_classifier.k
            posterior_probs[label] = posterior_prob

            # Calculate distance to predicted class
            min_distance = np.min(label_distances[:ki])
            distances[label] = min_distance

        # Determine the majority label among the k-nearest neighbors
        majority_label = max(posterior_probs, key=posterior_probs.get)

        if posterior_probs[majority_label] > self.min_prob:
            # Return the predicted label, posterior probability, and distance
            return majority_label, posterior_probs, distances[majority_label]


# The FaceClustering class enables unsupervised clustering of face images according to their identity and
# re-identification.
class FaceClustering:

    # Prepare FaceClustering; specify all parameters of clustering algorithm.
    def __init__(self,num_clusters=2, max_iter=25):
        # ToDo: Prepare FaceNet.
        self.facenet = FaceNet()

        # The underlying gallery: embeddings without class labels.
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))

        # Number of cluster centers for k-means clustering.
        self.num_clusters = num_clusters
        # Cluster centers.
        self.cluster_center = np.empty((num_clusters, self.facenet.get_embedding_dimensionality()))
        # Cluster index associated with the different samples.
        self.cluster_membership = []

        # Maximum number of iterations for k-means clustering.
        self.max_iter = max_iter

        # Load face clustering from pickle file if available.
        if os.path.exists("clustering_gallery.pkl"):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open("clustering_gallery.pkl", 'w') as f:
            pickle.dump((self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership), f)

    # Load trained model from a pickle file.
    def load(self):
        with open("clustering_gallery.pkl", 'r') as f:
            (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = pickle.load(f)

    # ToDo
    def update(self, face):
        embedding = self.facenet.predict(face)
        self.embeddings = np.vstack((self.embeddings, embedding))
        self.save()

    
    # ToDo
    def fit(self):
        if len(self.embeddings) >= self.num_clusters:
            kmeans = KMeans(n_clusters=self.num_clusters, max_iter=self.max_iter, random_state=42)
            kmeans.fit(self.embeddings)
            self.cluster_center = kmeans.cluster_centers_
            self.cluster_membership = kmeans.predict(self.embeddings)
            self.save()


    # ToDo
    def predict(self, face):
        if len(self.embeddings) == 0:
            return None

        embedding = self.facenet.predict(face)

        # Calculate distances to cluster centers using a brute-force method
        distances = np.linalg.norm(self.cluster_center - embedding, axis=1)

        # Find the closest cluster
        closest_cluster = np.argmin(distances)

        return closest_cluster




# def test_face_recognition():
#     # Create an instance of FaceRecognizer
#     face_recognizer = FaceRecognizer()
#
#     # Load an image for testing (replace 'your_image_path.jpg' with the actual path)
#     image_path = 'your_image_path.jpg'
#     image = cv2.imread(image_path)
#
#     # Perform face recognition and update the model
#     label = "Person1"
#     face_recognizer.update(image, label)
#
#     # Save the model (optional)
#     face_recognizer.save()
#
#     # Perform prediction on the same image
#     predicted_label, confidence = face_recognizer.predict(image)
#
#     if predicted_label is not None:
#         print(f"Predicted Label: {predicted_label}, Confidence: {confidence}")
#     else:
#         print("No prediction")
#
# def test_face_clustering():
#     # Create an instance of FaceClustering
#     face_clustering = FaceClustering()
#
#     # Load an image for testing (replace 'your_image_path.jpg' with the actual path)
#     image_path = 'your_image_path.jpg'
#     image = cv2.imread(image_path)
#
#     # Perform face clustering and update the model
#     face_clustering.update(image)
#
#     # Save the model (optional)
#     face_clustering.save()
#
#     # Fit the clustering model
#     face_clustering.fit()
#
#     # Perform prediction on the same image
#     predicted_cluster = face_clustering.predict(image)
#
#     if predicted_cluster is not None:
#         print(f"Predicted Cluster: {predicted_cluster}")
#     else:
#         print("No prediction")
#
# # Call the test functions to see the results
# test_face_recognition()
# test_face_clustering()
