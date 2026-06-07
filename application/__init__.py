from flask import Flask, request, Response, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# -----------------------------
# 1. LOAD & PREPARE DATA
# -----------------------------
df = pd.read_csv("./studentData/students.csv")

# Separate features and target
X = df.drop(columns=['Target'])
y = df['Target']

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_scaled, y_encoded)

# Save model + scaler + label encoder
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

# -----------------------------
# 2. FLASK APP
# -----------------------------
app = Flask(__name__)

@app.route('/api', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Extract numeric features in correct order (matching your CSV)
    input_data = np.array([
        data["course"],
        data["sneeds"],
        data["debtor"],
        data["tuition"],
        data["gender"],
        data["scholarship"],
        data["age"],
        data["international"],
        data["first_enrolled"],
        data["first_approved"],
        data["second_enrolled"],
        data["second_approved"]
    ]).reshape(1, -1)

    # Load preprocessors + model
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    result = label_encoder.inverse_transform(prediction)[0]

    return Response(json.dumps(result))

if __name__ == '__main__':
    app.run()
