import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Auto-run training if model artifacts don't exist yet
if not os.path.exists("models/crop_yield_artifacts.pkl"):
    import train_and_evaluate
    train_and_evaluate.main()

app = Flask(__name__)

artifacts = joblib.load("models/crop_yield_artifacts.pkl")
with open("models/metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)


@app.route("/")
def index():
    return render_template("index.html", metadata=metadata)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() or request.form

        # Support both Categorical Name OR Numeric Label Code (as shown in Paper Fig. 4 & 5: Area=85, Item=4)
        area_input = str(data.get("area", "India")).strip()
        item_input = str(data.get("item", "Potatoes")).strip()

        area_map = artifacts["area_mapping"]
        item_map = artifacts["item_mapping"]
        code_to_area = artifacts["code_to_area"]
        code_to_item = artifacts["code_to_item"]

        if area_input.isdigit():
            area_code = int(area_input)
            area_code = min(max(0, area_code), len(area_map) - 1)
            area_name = code_to_area.get(area_code, "India")
        else:
            area_name = area_input if area_input in area_map else "India"
            area_code = area_map[area_name]

        if item_input.isdigit():
            item_code = int(item_input)
            item_code = min(max(0, item_code), len(item_map) - 1)
            item_name = code_to_item.get(item_code, "Potatoes")
        else:
            item_name = item_input if item_input in item_map else "Potatoes"
            item_code = item_map[item_name]

        year = int(float(data.get("year", 2008)))
        rainfall = float(data.get("rainfall", 1120.0))
        pesticides = float(data.get("pesticides", 40719.0))
        temperature = float(data.get("temperature", 25.8))
        selected_model = data.get("model", "Voting")

        features_df = pd.DataFrame(
            [[area_code, item_code, year, rainfall, pesticides, temperature]],
            columns=[
                "Area_Code",
                "Item_Code",
                "Year",
                "average_rain_fall_mm_per_year",
                "pesticides_tonnes",
                "avg_temp",
            ],
        )

        # Predict with all 4 models so the user gets the primary outcome + side-by-side comparison
        features_scaled = artifacts["scaler_global"].transform(features_df)

        pred_voting = float(artifacts["voting_model"].predict(features_df)[0])
        pred_gb = float(artifacts["gb_model"].predict(features_df)[0])
        pred_knn = float(artifacts["knn_model"].predict(features_scaled)[0])
        pred_lr = float(artifacts["lr_model"].predict(features_scaled)[0])

        model_outputs = {
            "Voting": round(pred_voting, 4),
            "GradientBoosting": round(pred_gb, 4),
            "KNN": round(pred_knn, 4),
            "LinearRegression": round(pred_lr, 4),
        }

        primary_yield_hg_ha = model_outputs.get(selected_model, model_outputs["Voting"])
        yield_tonnes_ha = round(primary_yield_hg_ha / 10000.0, 2)

        return jsonify(
            {
                "status": "success",
                "area_name": area_name,
                "area_code": area_code,
                "item_name": item_name,
                "item_code": item_code,
                "year": year,
                "rainfall": rainfall,
                "pesticides": pesticides,
                "temperature": temperature,
                "selected_model": selected_model,
                "predicted_yield_hg_ha": primary_yield_hg_ha,
                "predicted_yield_tonnes_ha": yield_tonnes_ha,
                "all_model_predictions": model_outputs,
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  🚀 CROP YIELD PREDICTION SERVER RUNNING AT: http://127.0.0.1:5000")
    print("=" * 65 + "\n")
    app.run(debug=True, port=5000)

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)