from flask import Flask, render_template, request
import pandas as pd
import joblib


app = Flask(__name__)

# Load saved model and preprocessing objects
model = joblib.load("silent_sting.pkl")
features = joblib.load("sting_features.pkl")
numeric_cols = joblib.load("sting_numeric_cols.pkl")
scaler = joblib.load("sting_scaler.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        gender = request.form["Gender"]
        time_since_bite = float(request.form["Time_Since_Bite_Min"])
        heart_rate = float(request.form["Heart_Rate_BPM"])
        blood_pressure = float(request.form["Blood_Pressure_Systolic"])
        local_swelling = request.form["Local_Swelling"]
        muscle_paralysis = int(request.form["Muscle_Paralysis_Present"])
        blood_coagulation = int(request.form["Blood_Coagulation_Failure"])

        # Same encoding used during training
        gender_mapping = {
            "Male": 0,
            "Female": 1,
            "Other": 2
        }

        swelling_mapping = {
            "None": 0,
            "Mild": 1,
            "Medium": 2,
            "Severe": 3
        }

        gender = gender_mapping[gender]
        local_swelling = swelling_mapping[local_swelling]

        # Create input data
        input_data = pd.DataFrame([{
            "Gender": gender,
            "Time_Since_Bite_Min": time_since_bite,
            "Heart_Rate_BPM": heart_rate,
            "Blood_Pressure_Systolic": blood_pressure,
            "Local_Swelling": local_swelling,
            "Muscle_Paralysis_Present": muscle_paralysis,
            "Blood_Coagulation_Failure": blood_coagulation
        }])

        # Ensure exact feature order
        input_data = input_data[features]

        # Scale numeric features
        input_data[numeric_cols] = scaler.transform(
            input_data[numeric_cols]
        )

        
        
        # Prediction
        result = model.predict(input_data)

        class_labels = {
        0: "Black Widow Spider",
        1: "Harmless Insect",
        2: "Scorpion",
        3: "Viper Snake"
        }

        prediction = class_labels[int(result[0])]

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)