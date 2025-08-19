from flask import Flask, request, render_template
import pandas as pd
from pickle import load

# Load model, scaler and columns
f = open("cp.pkl", "rb")
model = load(f)
f.close()

f = open("scaler.pkl", "rb")
scaler = load(f)
f.close()

f = open("columns.pkl", "rb")
columns = load(f)
f.close()
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    price = None
    if request.method == "POST":
        # Collect form inputs
        car_name = request.form.get("car_name")
        brand = request.form.get("brand")
        car_model = request.form.get("car_model")
        vehicle_age = float(request.form.get("vehicle_age"))
        km_driven = float(request.form.get("km_driven"))
        seller_type = request.form.get("seller_type")
        fuel_type = request.form.get("fuel_type")
        transmission_type = request.form.get("transmission_type")
        mileage = float(request.form.get("mileage"))
        engine = float(request.form.get("engine"))
        max_power = float(request.form.get("max_power"))
        seats = int(request.form.get("seats"))

        # Create dataframe
        car = {
            "car_name": car_name,
            "brand": brand,
            "model": car_model,
            "vehicle_age": vehicle_age,
            "km_driven": km_driven,
            "seller_type": seller_type,
            "fuel": fuel_type,
            "transmission": transmission_type,
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "seats": seats
        }
        car_df = pd.DataFrame([car])

        # One-hot encode + align columns
        car_dummies = pd.get_dummies(car_df)
        car_dummies = car_dummies.reindex(columns=columns, fill_value=0)

        # Scale
        car_scaled = scaler.transform(car_dummies)

        # Predict
        price = model.predict(car_scaled)[0]
        price = max(price, 0)

    return render_template("home.html", price=price)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)

