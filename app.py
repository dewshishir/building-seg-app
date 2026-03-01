import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model.seg_model import SegModel
# ------------------ Load Model ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = SegModel.load_from_checkpoint("model/model.ckpt")
    model.to(device)
    model.eval()
    return model

model = load_model()

# ------------------ Prediction ------------------
def predict(image):

    original = np.array(image)
    image_resized = cv2.resize(original, (512,512))

    tensor = torch.tensor(image_resized).permute(2,0,1).float()/255.0
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(tensor)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()

    return original, pred.squeeze().cpu().numpy()

# ------------------ UI ------------------
st.title("Building Segmentation Demo")

image_file = st.file_uploader("Upload Image", type=["png", "jpg"])
mask_file = st.file_uploader("Upload Ground Truth Mask (Optional)", type=["png"])

if image_file:

    image = Image.open(image_file).convert("RGB")
    original, pred_mask = predict(image)

    col1, col2, col3 = st.columns(3)

    col1.image(original, caption="Original")

    if mask_file:
        gt = Image.open(mask_file).convert("L")
        gt = np.array(gt.resize((512,512))) / 255.0
        col2.image(gt, caption="Ground Truth")

    col3.image(pred_mask, caption="Prediction")