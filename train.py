import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("custom_gesture_dataset.csv")

# Split data into features and labels
X = data.drop("label", axis=1).values  # Landmark coordinates
y = data["label"].values  # Gesture labels

# Encode string labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict and decode predictions back to original labels
y_pred_encoded = clf.predict(X_test)
y_pred = le.inverse_transform(y_pred_encoded)

# Decode y_test for comparison
y_test_decoded = le.inverse_transform(y_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test_decoded, y_pred):.2f}")
