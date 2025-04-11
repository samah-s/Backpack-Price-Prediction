import streamlit as st
from PIL import Image
import joblib
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

# Load model and preprocessors
model = joblib.load("artifact/model.pkl")
indexers = joblib.load("artifact/indexers.pkl")
columns = joblib.load("artifact/columns.pkl")
standard_scaler = joblib.load("artifact/scaler_std.pkl")
minmax_scaler = joblib.load("artifact/scaler_minmax.pkl")

# Load CLIP model
@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

clip_model, clip_processor = load_clip_model()

st.title("🎒 Backpack Price Predictor ")

uploaded_file = st.file_uploader("Upload an image of a backpack", type=["jpg", "jpeg", "png"])

# Reference samples (you'll need to store representative images for each class)
allowed_values = {
    "Brand": ["Nike", "Jansport", "Puma", "Under Armour"],
    "Material": ["Polyester", "Nylon", "Leather"],
    "Size": ["Small", "Medium", "Large"],
    "Laptop Compartment": ["Yes", "No"],
    "Waterproof": ["Yes", "No"],
    "Style": ["Backpack"],
    "Color": ["Black", "Blue", "Pink","Red", "Green", "Grey"]
}

# You should place reference images in a folder like 'ref_images/<Attribute>/<Value>.jpg'
def get_best_match(image, attribute):
    texts = [f"{value} backpack" for value in allowed_values[attribute]]
    inputs = clip_processor(text=texts, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).detach().numpy()[0]
    return allowed_values[attribute][np.argmax(probs)]

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.subheader("🧠 Predicting Features from Image (using CLIP)")

    extracted = {
        "Weight Capacity (kg)": 7.5 if get_best_match(image, "Size")=="Small" else (16.5 if get_best_match(image, "Size")=="Medium" else 24.0),
        "Brand": get_best_match(image, "Brand"),
        "Material": get_best_match(image, "Material"),
        "Size": get_best_match(image, "Size"),
        "Compartments": 2 if get_best_match(image, "Size")=="Small" else (4 if get_best_match(image, "Size")=="Medium" else 6),
        "Laptop Compartment": get_best_match(image, "Laptop Compartment"),
        "Waterproof": get_best_match(image, "Waterproof"),
        "Style": get_best_match(image, "Style"),
        "Color": get_best_match(image, "Color"),
    }

    st.subheader("📋 Extracted Features")
    st.json(extracted)

    # Same preprocessing and prediction pipeline as before
    df = pd.DataFrame([extracted])
    df["Weight Capacity (kg)"] = df["Weight Capacity (kg)"].astype(np.float32)

    num_cols = ["Weight Capacity (kg)", "Compartments"]
    df[num_cols] = standard_scaler.transform(df[num_cols])
    df[num_cols] = minmax_scaler.transform(df[num_cols])
    scaled_weight = df["Weight Capacity (kg)"].iloc[0]

    for prec in [7, 8, 9]:
        df[f"round{prec}"] = round(scaled_weight, prec)

    fractional = str(scaled_weight).split(".")[1] if "." in str(scaled_weight) else "0"
    fractional = (fractional + "0000")[:4]
    for k in range(1, 5):
        df[f'digit{k}'] = int(fractional[k - 1])

    onehot_features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
    df = pd.get_dummies(df, columns=onehot_features)

    for col in columns:
        if col not in df.columns:
            df[col] = 0
    df = df[columns]

    CATS = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
    for i, c1 in enumerate(CATS[:-1]):
        for c2 in CATS[i + 1:]:
            key = f"{c1}_{c2}"
            if key in indexers:
                pair_val = df[c1].astype(str) + "_" + df[c2].astype(str)
                df[key] = indexers[key].get_indexer(pair_val).astype(np.int16)
            else:
                df[key] = 0

    for col in columns:
        if col not in df.columns:
            df[col] = 0

    df = df[columns]
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    predicted_price = model.predict(df)[0]
    st.success(f"💰 **Predicted Price**: ${predicted_price:.2f}")
