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
    image = Image.open(BytesIO(bytes_data)).convert("RGB")
    draw = ImageDraw.Draw(image)
    start = time.time()
    boxes, probs = model(image)
    end = time.time()
    if len(probs) != 0:
        faces = 0
        for box, prob in zip(boxes, probs):
            if prob > 0.95:
                faces += 1
                draw.rectangle(box.tolist(), outline='red', width=2)
        st.image(image, caption=f"{faces} face(s) detected")
        st.write(f"Model latency: {(end - start):.5f} s")
    else:
        st.write("No faces detected")
