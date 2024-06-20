import cv2
import numpy as np
import pickle
import os
import random
from sklearn.cluster import KMeans
from face_detector import FaceDetector

import matplotlib.pyplot as plt

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

    # Get dimensionality of the extracted embeddings.
    def get_embedding_dimensionality(self):
        return self.dim_embeddings
    
def get_key_from_value(dictionary, target_value):
        for key, value in dictionary.items():
            if value == target_value:
                return key
        return None

# The FaceRecognizer model enables supervised face identification.
class FaceRecognizer:

    # Prepare FaceRecognizer; specify all parameters for face identification.
    def __init__(self, num_neighbours=11, max_distance=0.8, min_prob=0.5):
        # ToDo: Prepare FaceNet and set all parameters for kNN.
        self.facenet = FaceNet()
        self.num_neighbours = num_neighbours
        self.max_distance = max_distance
        self.min_prob = min_prob
        # The underlying gallery: class labels and embeddings.
        self.labels = []
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))
        self.classifier= NearestNeighborClassifier()
        self.label_dict = {"Alan Ball":0,"Manuel Pellegrini":1,"Marina Silva":2,"Nancy Sinatra":3,"Peter Gilmour":4}

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
        self.embeddings = np.append(self.embeddings, [self.facenet.predict(face)], axis=0) 
         
        self.labels.append(self.label_dict[label])
        #self.nearest_neighbor_classifier.fit(self.embeddings, np.array(self.labels))
        self.classifier.fit(self.embeddings,np.array(self.labels))
        self.save()
        
    # ToDo
    def predict(self, face):
        # identity is determined by taking the majority of identities of the k nearest neighbors
        #model = KNeighborsClassifier(n_neighbors=self.num_neighbours)
        embeddings = self.facenet.predict(face)
        embeddings = embeddings.reshape(1,-1)
        
        labels = np.array(self.labels)
        unique_labels = list(set(labels))
        self.classifier.fit(self.embeddings, labels)
        posterior_probabilities = [0] * len(set(self.labels))
        self.classifier.set_k_neighbours(self.num_neighbours)
        predictions,vector_result,similiarities = self.classifier.predict_labels_and_similarities(embeddings)
        #distance, index = model.kneighbors(embeddings,self.num_neighbours)
        #print(similiarities)
        #print(similiarities.shape)
        #print(predictions)
        #print(predictions.shape)
        #print(retval)
        #print(vector_result)
        predicted_label = int(predictions[0])
        #posterior_probabilities = np.zeros(len(self.class_labels))  # Assuming self.class_labels contains unique class labels
        for i, class_label in enumerate(unique_labels):
            ki = np.count_nonzero(vector_result == class_label)
            posterior_probabilities[i] = ki / self.num_neighbours
        posterior_prob = posterior_probabilities[predicted_label]
        distance_to_class = min(similiarities.flatten())
        #probability = model.predict_proba(embeddings)[0]
        #print(len(posterior_probabilities))
        #print(predicted_label,posterior_prob,distance_to_class)
        predicted_person = get_key_from_value(self.label_dict,predicted_label)
        
        # Open Set protocol
        if posterior_prob < self.min_prob or distance_to_class > self.max_distance:
            predicted_person = "Unknown"
        return predicted_person, posterior_prob, distance_to_class



# The FaceClustering class enables unsupervised clustering of face images according to their identity and
# re-identification.
class FaceClustering:

    # Prepare FaceClustering; specify all parameters of clustering algorithm.
    def __init__(self,num_clusters=3, max_iter=25):
        # ToDo: Prepare FaceNet.
        self.facenet = FaceNet()
        # The underlying gallery: embeddings without class labels.
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))

        # Number of cluster centers for k-means clustering.
        self.num_clusters = num_clusters
        # Cluster centers.
        self.cluster_centers = np.empty((num_clusters, self.facenet.get_embedding_dimensionality()))
        # Cluster index associated with the different samples.
        self.cluster_membership = []
        self.k_mean_inertia=[]
        # Maximum number of iterations for k-means clustering.
        self.max_iter = max_iter

        # Load face clustering from pickle file if available.
        if os.path.exists("clustering_gallery.pkl"):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open("clustering_gallery.pkl", 'wb') as f:
            pickle.dump((self.embeddings, self.num_clusters, self.cluster_centers, self.cluster_membership), f)

    # Load trained model from a pickle file.
    def load(self):
        with open("clustering_gallery.pkl", 'rb') as f:
            (self.embeddings, self.num_clusters, self.cluster_centers, self.cluster_membership) = pickle.load(f)

    # ToDo
    def update(self, face):
        embedding = self.facenet.predict(face)
        self.embeddings = np.append(self.embeddings, [embedding], axis=0)
        self.fit()
        #print(self.cluster_membership)
        self.save()
        
    # ToDo implement k-means clustering without using any ml library
    def fit(self):
        # Initialize cluster centers randomly
        self.cluster_centers = self.embeddings[np.random.choice(len(self.embeddings), self.num_clusters, replace=True)]

        for _ in range(self.max_iter):
            # Assign each sample to the nearest cluster
            distances = np.linalg.norm(self.embeddings[:, np.newaxis] - self.cluster_centers, axis=2)
            labels = np.argmin(distances, axis=1)

            # Update cluster centers
            new_centers = np.array([self.embeddings[labels == i].mean(axis=0) for i in range(self.num_clusters)])

            # Check for convergence
            if np.allclose(new_centers, self.cluster_centers, rtol=1e-4):
                break

            self.cluster_centers = new_centers

            # Calculate k-means inertia (sum of squared distances to nearest cluster center)
            inertia = np.sum(np.min(distances, axis=1))
            self.k_mean_inertia.append(inertia)
            
        print(labels)
        plt.plot(range(1, len(self.k_mean_inertia) + 1), self.k_mean_inertia, marker='o')
        plt.show()
        self.cluster_membership = labels


        
    # ToDo
    def predict(self, face):
        # Assign the face to the nearest cluster
        embedding = self.facenet.predict(face).reshape(1, -1)
        distances = np.linalg.norm(embedding - self.cluster_centers, axis=1)
        predicted_cluster = np.argmin(distances)
        minimum_distance = np.min(distances)
        
        return predicted_cluster,minimum_distance
    '''
    def predict(self, face):
        embedding = self.facenet.predict(face)
        distance= list()
        for i in range(self.num_clusters):
            distance.append(np.linalg.norm(embedding-self.cluster_centers[i]))
            
        matching_index = distance.index(min(distance))
        
        return matching_index,distance 
    '''
