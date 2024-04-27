from models import facerec, embedder

from PIL import Image
import streamlit as st
from io import BytesIO
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import time

st.set_page_config(page_title="Bringer - Face Grouping Test")


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

# Get the epsilon, min_face_size and the images
epsilon = st.slider("Epsilon", min_value=0.1, max_value=0.9, step=0.01, value=0.23)
min_face_size = st.slider("Minimum face size", min_value=1, max_value=30, step=1, value=20)
min_samples = st.slider("Minimum samples", min_value=1, max_value=10, step=1, value=1)
uploaded = st.file_uploader("Select a photo", type=['png', 'jpg'], accept_multiple_files=True)

images = []
if len(uploaded) > 0:
    # Detect the faces in the images
    result = []
    compression_progress = st.progress(0, text="Compressing images...")
    n_images = len(uploaded)
    for i, image in enumerate(uploaded):
        bytes_data = image.getvalue()
        bytes_file = BytesIO(bytes_data)
        if image.size / 1024 > 500:
            old_image = Image.open(bytes_file).convert("RGB")
            new_size = old_image.size[0] // 4, old_image.size[1] // 4
            resized_image = old_image.resize(new_size)
            old_image.close()
            buff = BytesIO()
            resized_image.save(buff, format='PNG', optimize=True)
            resized_image.close()
            bytes_file = buff
        image = Image.open(bytes_file).convert("RGB")
        images.append(np.array(image))
        result.extend(facerec_model.detect([image], minsize=min_face_size))
        compression_progress.progress(int(((i + 1) / n_images)*100), text="Compressing images...")
    time.sleep(1)
    compression_progress.empty()

    start = time.time()
    faces = []
    with st.spinner("Detecting faces..."):
        for i, res in enumerate(result):
            if res is None:
                continue
            # extract faces
            boxes, probs, lands = res
            for j, box in enumerate(boxes):
                # confidence of detected face
                if probs[j] > 0.98:
                    h, w = images[i].shape[:2]
                    x1, y1, size = get_boundingbox(box, w, h)
                    face = images[i][y1:y1+size, x1:x1+size]
                    faces.append(face)

    groups = {}
    with st.spinner("Clustering the faces..."):
        # Group the faces
        embeddings = embedder_model.embedding(faces)
        dbscan = DBSCAN(eps=epsilon, metric='cosine', min_samples=min_samples)
        labels = dbscan.fit_predict(embeddings)

        # Show the faces along with the labels
        for face, label in zip(faces, labels):
            if label not in groups:
                groups[label] = []
            groups[label].append(face)
    end = time.time()

    st.text(f"Processed {len(faces)} faces in {end - start} seconds.")

    for group in groups.keys():
        plt.figure(figsize=(3 * len(groups[group]), 2))
        st.text(f'Class {group}')
        plt.axis('off')
        for i, face in enumerate(groups[group]):
            plt.subplot(1, len(groups[group]), i + 1)
            plt.imshow(face)
            plt.axis('off')
        plt.savefig("faces.png")
        plt.close()
        st.image("faces.png")
