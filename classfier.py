import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset, skipping any malformed lines
data = pd.read_csv("custom_gesture_dataset.csv", on_bad_lines='skip')

# Step 2: Preprocess the data
# Assuming 'label' is the column with gesture labels, and all other columns are feature coordinates
X = data.drop("label", axis=1).values  # Features (landmark coordinates)
y = data["label"].values  # Labels (gesture categories)

# Step 3: Encode labels if they are strings (e.g., "thumbs_up", "peace")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Encode labels as integers

# Step 4: Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 5: Initialize the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 6: Train the model
clf.fit(X_train, y_train)

# Step 7: Make predictions on the test set
y_pred = clf.predict(X_test)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Step 9: Decode the predicted labels and original labels back to string form
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_test_labels = label_encoder.inverse_transform(y_test)

# Print some example predictions
print("Example predictions:")
for true, pred in zip(y_test_labels[:5], y_pred_labels[:5]):
    print(f"True: {true} | Predicted: {pred}")
