import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from tensorflow.keras.models import load_model  # Ensure you have the correct model path
import matplotlib.pyplot as plt

# Load your trained model
model = load_model("C:\\Users\\ADMIN\\Downloads\\mnist_ann_model.keras")  



st.title("🧠 Handwritten Digit Recognizer")
st.write("Draw a digit (0-9) below. The model will try to predict what you wrote!")

# Canvas settings
canvas_result = st_canvas(
    fill_color="white",  # Color for new strokes
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Resize image and convert to grayscale
    img = canvas_result.image_data
    img_gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    img_resized = cv2.resize(img_gray, (28, 28))
    img_inverted = 255 - img_resized
    img_norm = img_inverted / 255.0
    img_input = img_norm.reshape(1, 28, 28)  # ✅ Correct shape
  

    st.subheader("🖼️ Preview")
    st.image(img_resized, width=150, clamp=True, channels="L", caption="28x28 input")

    if st.button("🔍 Predict"):
        pred = model.predict(img_input)
        pred_digit = np.argmax(pred)

        st.success(f"✅ Predicted Digit: **{pred_digit}**")
        st.subheader("📊 Prediction Probabilities")
        st.bar_chart(pred[0])

