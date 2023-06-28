import skimage
import sklearn.metrics
from sklearn.cluster import KMeans
from rembg import remove
from PIL import Image
import os
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
import numpy as np

def elbow_graph(data):
    x = []
    y = []
    for i in range(1, 20):
        kmeans = KMeans(n_clusters=i, n_init="auto").fit(data)
        x.append(i)
        y.append(kmeans.inertia_)
    plt.plot(x, y)
    plt.title('Elbow graph')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.savefig("elbow_graph.png")


def confusion_matrix(results):
    matrix = np.zeros((10,10))
    for i in range(1000):
        matrix[i//100][results[i]]+=1
    print(matrix)
    alpha = ['Plane', 'Bonsai tree', 'Car', 'Chandelier', 'Piano', 'Turtle', 'Sailboat', 'Leopard', 'Motorbike',
             'Watch']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.matshow(matrix, cmap=plt.cm.Blues)
    xaxis = np.arange(len(alpha))
    ax.set_yticks(xaxis)
    ax.set_yticklabels(alpha)
    plt.show()



if __name__ == "__main__":
    path = "Cluster_img"
    if len(sys.argv)<=1:
        # Preprocessing - remove background and resize images
        if not os.path.exists("Preprocessed"):
            os.makedirs("Preprocessed")
        for filename in os.listdir(path):
            f = os.path.join(path,filename)
            img=Image.open(f)
            x, y = img.size
            output = remove(img)
            new_img = Image.new('RGBA', (512, 512), (0,0,0,0))
            new_img.paste(output, (int((512 - x) / 2), int((512 - y) / 2)))
            output_path = os.path.join("Preprocessed", os.path.splitext(filename)[0] + ".png")
            new_img.save(output_path)
            path = "Preprocessed"
    else:
        path = str(sys.argv[1])
    data = []
    # Feature extraction - HOG
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        img = cv2.imread(f)  # Read in the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # and convert to grayscale
        data.append(skimage.feature.hog(gray))
    # Dimensionality reduction
    pca = PCA(n_components=16)
    pca_data = pca.fit_transform(data)
    labels = []
    results = []
    for i in range(1000):
        labels.append(i//100)
    kmeans = KMeans(n_clusters=10, n_init="auto").fit(pca_data)
    print("Accuracy",sklearn.metrics.adjusted_rand_score(labels, kmeans.labels_))
    confusion_matrix(kmeans.labels_)
    elbow_graph(pca_data)