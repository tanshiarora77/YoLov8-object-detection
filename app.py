import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# Load YOLO model
model = YOLO("best.pt")

st.title("😷 Face Mask Detection using YOLOv8")
st.write("Upload an image to detect face masks.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)

        results = model.predict(tmp.name, conf=0.25)

        result_image = results[0].plot()

        st.image(
            result_image,
            caption="Detection Result",
            use_container_width=True
        )