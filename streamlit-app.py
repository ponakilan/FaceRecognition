import io

from models.mtcnn import MTCNN
from PIL import Image, ImageDraw
import streamlit as st
from io import BytesIO
import time

st.title("Detect Faces")
min_face_size = st.slider("Minimum face size", min_value=1, max_value=30, step=1, value=20)

model = MTCNN(weights_path="models/TrainedWeights", min_face_size=min_face_size).eval()

uploaded_file = st.file_uploader("Select a photo", type=['png', 'jpg'])
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    bytes_file = BytesIO(bytes_data)
    if uploaded_file.size / 1024 > 500:
        old_image = Image.open(bytes_file).convert("RGB")
        new_size = old_image.size[0]//4, old_image.size[1]//4
        resized_image = old_image.resize(new_size)
        old_image.close()
        buff = io.BytesIO()
        resized_image.save(buff, format='PNG', optimize=True)
        resized_image.close()
        bytes_file = buff
    image = Image.open(bytes_file).convert("RGB")
    draw = ImageDraw.Draw(image)
    start = time.time()
    boxes, probs = model(image)
    end = time.time()
    if len(probs) != 0:
        faces = 0
        for box, prob in zip(boxes, probs):
            if prob > 0.9:
                faces += 1
                draw.rectangle(box.tolist(), outline='red', width=4)
        st.image(image, caption=f"{faces} face(s) detected")
        image.close()
        del draw
        st.write(f"Model latency: {(end - start):.5f} s")
    else:
        st.write("No faces detected")
