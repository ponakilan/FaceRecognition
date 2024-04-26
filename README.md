# FaceRecognition

FaceRecognition is a lightweight face detection and verification module built for *Bringer - An AI based photo sharing platform*. <br>
It uses the MTCNN architecture trained on a custom dataset for face detection, and an InceptionResnet-based architecture to generate face embeddings for face verification and clustering. The InceptionResnet-based model is trained using `TripletMarginLoss` to generate face embeddings.

## Getting Started
This is an example of how you may give instructions on setting up your project locally. To get a local copy up and running follow these simple example steps.

1. Clone the repository
```
git clone https://github.com/ponakilan/FaceRecognition.git
```
2. Install the requirements
```
cd FaceRecognition
pip install -r requirements.txt
```
3. Run the streamlit server
```
streamlit run server.py
```
