from models import facerec, embedder

from PIL import Image
import streamlit as st
from io import BytesIO
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import math
import numpy as np
from collections import defaultdict


def get_boundingbox(box, w, h, scale=1.2):
    x1, y1, x2, y2 = box
    size = int(max(x2-x1, y2-y1) * scale)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    if size > w or size > h:
        size = int(max(x2-x1, y2-y1))
    x1 = max(int(center_x - size // 2), 0)
    y1 = max(int(center_y - size // 2), 0)
    size = min(w - x1, size)
    size = min(h - y1, size)
    return x1, y1, size


# Instantiate the models for face detection and embeddings
facerec_weights = "models/TrainedWeights/facerec.pt"
embedder_weights = "models/TrainedWeights/embedder.pt"

facerec_model = facerec.MTCNN(model=facerec_weights)
embedder_model = embedder.FaceNet(model=embedder_weights)

# Get the epsilon value and the images
epsilon = st.slider("Minimum face size", min_value=0.1, max_value=0.9, step=0.01, value=0.25)
uploaded = st.file_uploader("Select a photo", type=['png', 'jpg'], accept_multiple_files=True)

images = []
if len(uploaded) > 0:
    # Detect the faces in the images
    result = []
    for image in uploaded:
        bytes_data = image.getvalue()
        bytes_file = BytesIO(bytes_data)
        image = Image.open(bytes_file).convert("RGB")
        images.append(np.array(image))
        result.extend(facerec_model.detect([image]))

    faces = []
    for i, res in enumerate(result):
        if res is None:
            continue
        # extract faces
        boxes, probs, lands = res
        for j, box in enumerate(boxes):
            # confidence of detected face
            if probs[j] > 0.95:
                h, w = images[i].shape[:2]
                x1, y1, size = get_boundingbox(box, w, h)
                face = images[i][y1:y1+size, x1:x1+size]
                faces.append(face)

    # Group the faces
    embeddings = embedder_model.embedding(faces)
    dbscan = DBSCAN(eps=epsilon, metric='cosine', min_samples=1)
    labels = dbscan.fit_predict(embeddings)

    # Show the faces along with the labels
    n_rows = math.ceil(len(faces)/5)
    n_cols = math.ceil(len(faces)/n_rows)
    plt.figure(figsize=(16, 10))
    for i, face in enumerate(faces):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.title(labels[i])
        plt.imshow(face)
        plt.axis('off')
    plt.savefig("faces.png")
    st.image("faces.png")
