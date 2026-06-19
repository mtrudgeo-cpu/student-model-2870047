from flask import Flask, request, Response, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
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

# Split BEFORE scaling (best practice)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model 1: Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train_scaled, y_train)

# Train model 2: KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

 # Model_Evaluation 
rf_pred = rf.predict(X_test_scaled)
knn_pred = knn.predict(X_test_scaled)

rf_acc = accuracy_score(y_test, rf_pred)
knn_acc = accuracy_score(y_test, knn_pred)

print("Random Forest Accuracy:", rf_acc)
print("KNN Accuracy:", knn_acc)

# Confusion matrices
print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nKNN Confusion Matrix:")
print(confusion_matrix(y_test, knn_pred))


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
   # print("RAW DATA RECEIVED:", data) 

    # Extract numeric features in correct order (matching your CSV)
    input_data = np.array([
        int(data["course"]),
        int(data["sneeds"]),
        int(data["debtor"]),
        int(data["tuition"]),
        int(data["gender"]),
        int(data["scholarship"]),
        int(data["age"]),
        int(data["international"]),
        int(data["first_enrolled"]),
        int(data["first_approved"]),
        int(data["second_enrolled"]),
        int(data["second_approved"])
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
