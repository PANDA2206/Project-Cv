import cv2
import numpy as np
import pickle
import os
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from face_detector import FaceDetector


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
        self.labels.append(label)
        #self.nearest_neighbor_classifier.fit(self.embeddings, np.array(self.labels))
        self.save()

    # ToDo
    def predict(self, face):
        # identity is determined by taking the majority of identities of the k nearest neighbors
        model = KNeighborsClassifier(n_neighbors=self.num_neighbours)
        embeddings = self.facenet.predict(face)
        embeddings = embeddings.reshape(1,-1)
        print(embeddings.shape)
        
        labels = np.array(self.labels)
        model.fit(self.embeddings, labels)
        
        predictions = model.predict(embeddings)
        distance, index = model.kneighbors(embeddings,self.num_neighbours)
        
        print(distance)
        print(index.shape)
        
        probability = model.predict_proba(embeddings)[0]
        print(probability) 
        
        classes = model.classes_
        num_class = len(classes)
        print(num_class)
        class_dist = np.ones(num_class) * 100000
        #print(class_dist.shape)

        for i in index[0]:
            label = labels[i]
            class_index = np.where(classes == label)[0][0]
            if class_dist[class_index] > distance[0][class_index]:
                class_dist[class_index] = distance[0][class_index]
                
        # to find the matching class
        class_index = np.where(classes == predictions)[0][0]
        
        # Open Set protocol
        if probability[class_index] < self.min_prob and class_dist[class_index] > self.max_distance:
            predictions = "Unknown"
        return predictions, probability[class_index], class_dist[class_index]



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
        self.embeddings = np.append(self.embeddings, [embedding], axis=0)
        self.save()
        
    # ToDo
    def fit(self):
        k_mean=KMeans(n_clusters=self.num_clusters, max_iter=self.max_iter,random_state=42).fit(self.embeddings)

        self.cluster_centers=k_mean.cluster_centers_
        self.cluster_membership=k_mean.labels_

    # ToDo
    def predict(self, face):
        embedding = self.facenet.predict(face)
        distance= list()
        for i in range(self.num_clusters):
            distance.append(np.linalg.norm(embedding-self.cluster_centers[i]))
            
        matching_index = distance.index(min(distance))
        
        return matching_index,distance 
