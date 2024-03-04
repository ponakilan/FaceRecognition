from models.mtcnn import MTCNN
from PIL import Image, ImageDraw
import streamlit as st
from io import BytesIO

model = MTCNN(weights_path="models/TrainedWeights").eval()

st.title("Detect Faces")
uploaded_file = st.file_uploader("Select a photo", type=['png', 'jpg'])
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    image = Image.open(BytesIO(bytes_data)).convert("RGB")
    draw = ImageDraw.Draw(image)
    boxes, probs = model(image)
    faces = 0
    for box, prob in zip(boxes, probs):
        if prob > 0.9:
            faces += 1
            draw.rectangle(box.tolist(), outline='red', width=4)
    st.image(image, caption=f"{faces} face(s) detected")