import torch
import pandas as pd
import numpy as np
import joblib
from model import ImprovedNet
import argparse

def run_inference(input_path, model_path="models/best_model.pt", scaler_path="models/scaler.pkl"):
    data = pd.read_csv(input_path)
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(data)

    model = ImprovedNet(input_dim=X_scaled.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).numpy()
        preds = (probs > 0.5).astype(int).flatten()

    result = pd.DataFrame({
        "prediction": preds,
        "probability": probs.flatten()
    })
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV")
    args = parser.parse_args()

    result = run_inference(args.input)
    print(result.head())
