import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
data = pd.read_csv(
    r"C:\Users\hasin\OneDrive\Desktop\CODSOFT\Credit_Card_Fraud_Detection\creditcard.csv"
)

print("Dataset Loaded Successfully!")
print("\nFirst 5 Rows:")
print(data.head())

# Features and Target
X = data.drop("Class", axis=1)
y = data["Class"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Detailed Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)