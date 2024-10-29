########################################################################################################################
########################################################################################################################

import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Caricamento del modello ResNet50 pre-addestrato
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def load_image(image_path):
    # Carica un'immagine e ridimensionala
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # ResNet50 richiede immagini 224x224
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def extract_features(image_paths):
    # Estrae le caratteristiche visive dai tasselli
    features = []
    for image_path in image_paths:
        img = load_image(image_path)
        feature = model.predict(img)
        features.append(feature.flatten())
    return np.array(features)

# Esempio di utilizzo
image_dir = 'example_tiles/'  # Cartella con i tasselli
image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.png')]
features = extract_features(image_paths)


########################################################################################################################
########################################################################################################################

from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

print(features)
# Calcola le distanze Euclidee tra i tasselli
euclidean_dist = euclidean_distances(features)

# Calcola la somiglianza coseno tra i tasselli
cosine_sim = cosine_similarity(features)

########################################################################################################################

# Numero di cluster
n_clusters = 10

# Inizializza K-means e adatta il modello ai dati
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(features)

# Assegna ogni tassello a un cluster
labels = kmeans.labels_

# Visualizzazione dei risultati
def plot_clusters(image_paths, labels, n_clusters):
    for cluster in range(n_clusters):
        print(f"Cluster {cluster}:")
        plt.figure(figsize=(10, 10))
        idxs = np.where(labels == cluster)[0]
        for i, idx in enumerate(idxs[:9]):  # Visualizza fino a 9 immagini per cluster
            img = cv2.imread(image_paths[idx])
            plt.subplot(3, 3, i + 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
        plt.show()

# Visualizza i cluster
plot_clusters(image_paths, labels, n_clusters)

########################################################################################################################
########################################################################################################################
