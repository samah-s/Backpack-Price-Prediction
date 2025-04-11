import streamlit as st
from PIL import Image
import joblib
import pandas as pd
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load model and preprocessors

model = joblib.load("artifact/model.pkl")
indexers = joblib.load("artifact/indexers.pkl")
columns = joblib.load("artifact/columns.pkl")
standard_scaler = joblib.load("artifact/scaler_std.pkl")
minmax_scaler = joblib.load("artifact/scaler_minmax.pkl")

# Load BLIP model for feature extraction
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, blip_model = load_blip_model()

st.title("🎒 Backpack Price Predictor")

uploaded_file = st.file_uploader("Upload an image of a backpack", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Extract caption from image using BLIP
    st.subheader("🔍 Extracting Features using BLIP...")
    inputs = processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    st.write(f"**BLIP Caption**: _{caption}_")

    # Simple parsing - adjust as needed
    extracted = {
        "Weight Capacity (kg)": 24.0,  # Default or use some heuristic
        "Brand": "Nike" if "nike" in caption.lower() else "Unknown",
        "Material": "Polyester" if "polyester" in caption.lower() else "Nylon",
        "Size": "Medium",
        "Compartments": 2,
        "Laptop Compartment": "Yes" if "laptop" in caption.lower() else "No",
        "Waterproof": "Yes" if "waterproof" in caption.lower() else "No",
        "Style": "Backpack",
        "Color": "Black" if "black" in caption.lower() else "Blue"
    }

    st.subheader("📋 Extracted Features")
    st.json(extracted)

    # Create DataFrame
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

    # Categorical pair encodings
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

    # Predict
    predicted_price = model.predict(df)[0]
    st.success(f"💰 **Predicted Price**: ${predicted_price:.2f}")
