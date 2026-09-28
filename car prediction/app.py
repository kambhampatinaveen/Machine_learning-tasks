from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# --------------------------------------------------
# File paths
# --------------------------------------------------

MODEL_FILE = "car_price_model.pkl"
DATASET_FILE = "car_prediction_100.csv"


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        "car_price_model.pkl not found. "
        "Please train the model in your Jupyter Notebook first."
    )

model = joblib.load(MODEL_FILE)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(
        "car_prediction_100.csv not found."
    )

df = pd.read_csv(DATASET_FILE)


# --------------------------------------------------
# Get dropdown values from dataset
# --------------------------------------------------

brands = sorted(df["Brand"].dropna().unique().tolist())
models = sorted(df["Model"].dropna().unique().tolist())
fuel_types = sorted(df["Fuel_Type"].dropna().unique().tolist())
transmissions = sorted(df["Transmission"].dropna().unique().tolist())
colors = sorted(df["Color"].dropna().unique().tolist())


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html",
        brands=brands,
        models=models,
        fuel_types=fuel_types,
        transmissions=transmissions,
        colors=colors
    )


# --------------------------------------------------
# Prediction API
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Receive JSON data from frontend
        data = request.get_json()

        # --------------------------------------------------
        # Read values
        # --------------------------------------------------

        brand = data.get("Brand")
        model_name = data.get("Model")
        year = data.get("Year")
        kilometers = data.get("Kilometers_Driven")
        fuel_type = data.get("Fuel_Type")
        transmission = data.get("Transmission")
        previous_owners = data.get("Previous_Owners")
        engine_cc = data.get("Engine_CC")
        mileage = data.get("Mileage_KMPL")
        color = data.get("Color")

        # --------------------------------------------------
        # Validate input
        # --------------------------------------------------

        if not brand:
            return jsonify({
                "error": "Please select a brand."
            }), 400

        if not model_name:
            return jsonify({
                "error": "Please select a model."
            }), 400

        if not year:
            return jsonify({
                "error": "Please enter the manufacturing year."
            }), 400

        if not kilometers:
            return jsonify({
                "error": "Please enter kilometers driven."
            }), 400

        if not fuel_type:
            return jsonify({
                "error": "Please select fuel type."
            }), 400

        if not transmission:
            return jsonify({
                "error": "Please select transmission."
            }), 400

        if not previous_owners:
            return jsonify({
                "error": "Please enter previous owners."
            }), 400

        if not engine_cc:
            return jsonify({
                "error": "Please enter engine CC."
            }), 400

        if not mileage:
            return jsonify({
                "error": "Please enter mileage."
            }), 400

        if not color:
            return jsonify({
                "error": "Please select color."
            }), 400

        # --------------------------------------------------
        # Convert numerical values
        # --------------------------------------------------

        year = int(year)
        kilometers = float(kilometers)
        previous_owners = int(previous_owners)
        engine_cc = float(engine_cc)
        mileage = float(mileage)

        # --------------------------------------------------
        # Create DataFrame
        # --------------------------------------------------

        car_data = pd.DataFrame([{
            "Brand": brand,
            "Model": model_name,
            "Year": year,
            "Kilometers_Driven": kilometers,
            "Fuel_Type": fuel_type,
            "Transmission": transmission,
            "Previous_Owners": previous_owners,
            "Engine_CC": engine_cc,
            "Mileage_KMPL": mileage,
            "Color": color
        }])

        # --------------------------------------------------
        # Make prediction
        # --------------------------------------------------

        predicted_price = model.predict(car_data)[0]

        # Prevent negative price
        predicted_price = max(0, predicted_price)

        # --------------------------------------------------
        # Format price
        # --------------------------------------------------

        formatted_price = f"₹{predicted_price:,.0f}"

        # --------------------------------------------------
        # Send result to frontend
        # --------------------------------------------------

        return jsonify({
            "success": True,
            "predicted_price": round(float(predicted_price), 2),
            "formatted_price": formatted_price
        })

    except ValueError:

        return jsonify({
            "error": "Please enter valid numerical values."
        }), 400

    except Exception as e:

        print("Prediction Error:", e)

        return jsonify({
            "error": "Something went wrong while predicting the price."
        }), 500


# --------------------------------------------------
# Run Flask application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )