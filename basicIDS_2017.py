import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

# Load the data
path = input("Enter the path for training data: ") 
data = pd.read_csv(path)
print("Fetching the data...")

# Check for missing or infinite values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
if data.isnull().values.any():
    print("Missing or infinite values found. Filling with 0.")
    data.fillna(0, inplace=True)

# Split the data into training and testing sets
data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)

# Define the features and target
features = [
    ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets', 'Total Length of Fwd Packets',
    ' Total Length of Bwd Packets', ' Fwd Packet Length Max', ' Fwd Packet Length Min', 
    'Bwd Packet Length Max', ' Bwd Packet Length Min', ' Flow Packets/s',
    ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std',
    'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count', ' ACK Flag Count', ' Average Packet Size'
]
target = ' Label'

# Standardize the features
scaler = StandardScaler()
X_train = data_train[features]
y_train = data_train[target].apply(lambda x: 1 if x != 'BENIGN' else 0)

# Handle any remaining missing values (if they exist)
X_train.fillna(0, inplace=True)

X_train_scaled = scaler.fit_transform(X_train)

# Train the model
print("Training the model...")
mlp = MLPClassifier(hidden_layer_sizes=(26, 13), max_iter=5000, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Evaluate the model
X_test = data_test[features]
y_test = data_test[target].apply(lambda x: 1 if x != 'BENIGN' else 0)

# Handle any missing values in the test set (if they exist)
X_test.fillna(0, inplace=True)

X_test_scaled = scaler.transform(X_test)
predictions = mlp.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Function for making predictions on new samples
def predict(sample):
    sample_df = pd.DataFrame([sample], columns=features)
    sample_df.fillna(0, inplace=True)
    sample_scaled = scaler.transform(sample_df)
    prediction = mlp.predict(sample_scaled)
    return prediction[0]

# User interaction loop
while True:
    print("\nChoose an option:")
    print("1. Predict a single packet")
    print("2. Predict from a CSV file")
    print("3. Exit")
    choice = input("Enter your choice (1/2/3): ")

    if choice == '1':
        print(f"Enter the following in this format:\n {' '.join(features)}")
        sample_input = input()
        sample = [float(x) for x in sample_input.split()]
        result = predict(sample)
        if result == 1:
            print("Anomaly detected")
        elif result == 0:
            print("Normal packet")

    elif choice == '2':
        csv_path = input("Enter the path for the CSV file: ")
        new_data = pd.read_csv(csv_path)
        new_data['Prediction'] = new_data.apply(
            lambda row: "Anomaly" if predict(row[features].tolist()) == 1 else "Normal", axis=1)
        new_data.to_csv('predicted_data.csv', index=False)
        print("Predictions saved to 'predicted_data.csv'")

    elif choice == '3':
        print("Exiting...")
        break

    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
