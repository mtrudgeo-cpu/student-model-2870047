from flask import Flask, request, Response, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
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

# Train model 1: Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_scaled, y_encoded)

# Train model 2: KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y_encoded)

 # Model_Evaluation 
rf_pred = rf.predict(X_scaled)
knn_pred = knn.predict(X_scaled)
rf_acc = accuracy_score(y_encoded, rf_pred)
knn_acc = accuracy_score(y_encoded, knn_pred)

print("Random Forest Accuracy:", rf_acc)
print("KNN Accuracy:", knn_acc)

# Confusion matrices
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_encoded, rf_pred))

print("KNN Confusion Matrix:")
print(confusion_matrix(y_encoded, knn_pred))


# Save models + scaler + label encoder
joblib.dump(rf, "model.pkl")
joblib.dump(knn, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")


# -----------------------------
# 2. FLASK APP
# -----------------------------
app = Flask(__name__)

@app.route('/api', methods=['GET','POST'])
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

    # Load preprocessors 
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    # Scale input
    input_scaled = scaler.transform(input_data)

    #Choose model (default=random forest)
    model_choice = data.get("model", "rf")

    if model_choice == "knn":
        model=joblib.load("knn_model.pkl")
    else:
        model=joblib.load("model.pkl")

    # Predictions
    prediction = model.predict(input_scaled)
    original_label = label_encoder.inverse_transform(prediction)[0]

    #Map names
    label_map = {
    "Dropout": "High Risk",
    "Enrolled": "Medium Risk",
    "Graduate": "Low Risk"
    }
    
    # Convert model output to your custom label
    result = label_map.get(original_label, original_label)
    return Response(json.dumps(result))

if __name__ == '__main__':
    app.run()
