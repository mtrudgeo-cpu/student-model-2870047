from flask import Flask, request, Response, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack
import joblib

# -----------------------------
# 1. LOAD & PREPARE DATA
# -----------------------------
df = pd.read_csv(
    "./studentData/students.csv",
    header=0,
    names=[
        'course', 'sneeds', 'debtor', 'tuition', 'gender', 'scholarship',
        'age', 'international', 'first_enrolled', 'first_approved',
        'second_enrolled', 'second_approved', 'target'
    ]
)

y = df['target']

# Define column groups
numeric_cols = ['age', 'first_enrolled', 'first_approved', 'second_enrolled', 'second_approved']
categoric_cols = ['course', 'sneeds', 'debtor', 'tuition', 'gender', 'scholarship', 'international']

numeric_df = df[numeric_cols]
categoric_df = df[categoric_cols]

# -----------------------------
# 2. SCALE NUMERIC FEATURES
# -----------------------------
scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(numeric_df)

# -----------------------------
# 3. ONE-HOT ENCODE CATEGORICAL FEATURES
# -----------------------------
encoder = OneHotEncoder(handle_unknown='ignore')
categoric_encoded = encoder.fit_transform(categoric_df)

# -----------------------------
# 4. COMBINE FEATURES
# -----------------------------
X_final = hstack([categoric_encoded, numeric_scaled])

# -----------------------------
# 5. TRAIN MODEL
# -----------------------------
model = RandomForestClassifier(n_estimators=100)
model.fit(X_final, y)

# -----------------------------
# 6. SAVE PREPROCESSORS + MODEL
# -----------------------------
joblib.dump(model, "model.pkl")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

# -----------------------------
# 7. FLASK APP
# -----------------------------
app = Flask(__name__)

@app.route('/api', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Extract in correct order
    categoric_input = np.array([
        data["course"], data["sneeds"], data["debtor"], data["tuition"],
        data["gender"], data["scholarship"], data["international"]
    ]).reshape(1, -1)

    numeric_input = np.array([
        data["age"], data["first_enrolled"], data["first_approved"],
        data["second_enrolled"], data["second_approved"]
    ]).reshape(1, -1)

    # Load preprocessors + model
    encoder = joblib.load("encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("model.pkl")

    # Transform inputs
    categoric_encoded = encoder.transform(categoric_input)
    numeric_scaled = scaler.transform(numeric_input)

    # Combine
    X_request = hstack([categoric_encoded, numeric_scaled])

    # Predict
    prediction = model.predict(X_request)

    return Response(json.dumps(int(prediction[0])))

if __name__ == '__main__':
    app.run()
