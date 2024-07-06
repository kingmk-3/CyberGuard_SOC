#this uses the Nueral network with 2 hidden layres.
#Then grid search for number of nodes.
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

path = input("Enter the path for training data: ") 
data = pd.read_csv(path)
print("fetching the data...")

data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)
grouped_data = data_train.groupby(['protocol_type', 'service', 'flag'])
features = ['src_bytes', 'dst_bytes', 'duration', 'logged_in', 'count', 'srv_count',
            'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate']
target = 'class'
scaler = StandardScaler()
tuple_data = {}     #for different tuple of protocol, service and flag, we will have different model(they all be neural network)
for tuple_value, group in grouped_data:
    X = group[features]
    y = group[target].apply(lambda x: 1 if x != 'normal' else 0)
    X_scaled = scaler.fit_transform(X)
    tuple_data[tuple_value] = (X_scaled, y)
print("done...")

print("lets train our model...")
trained_models = {}
for tuple_value, (X_scaled, y) in tuple_data.items():
    mlp = MLPClassifier(hidden_layer_sizes=(26,13), max_iter=5000, random_state=42)
    mlp.fit(X_scaled, y)
    trained_models[tuple_value] = mlp
print("done...")
def predict(protocol_type, service, flag, sample): #this a bit different from the previous function that if first adds a name to each coloumns then proceeds
    tuple_value = (protocol_type, service, flag)
    model = trained_models.get(tuple_value)
    if model:
        sample_df = pd.DataFrame([sample], columns=features)
        sample_scaled = scaler.transform(sample_df)
        prediction = model.predict(sample_scaled)
        return prediction[0]
    else:
        return -1
print("lets check the accuracy of our model...")
test_grouped_data = data_test.groupby(['protocol_type', 'service', 'flag'])
true_values = []
predicted_values = []

for tuple_value, group in test_grouped_data:
    if tuple_value in trained_models:
        X_test = group[features]
        y_test = group[target].apply(lambda x: 1 if x != 'normal' else 0)
        X_test_scaled = scaler.transform(X_test)
        model = trained_models[tuple_value]
        predictions = model.predict(X_test_scaled)
        true_values.extend(y_test)
        predicted_values.extend(predictions)

accuracy = accuracy_score(true_values, predicted_values)
print(f'Accuracy: {accuracy * 100:.2f}%')

while True:
    print("\nChoose an option:")
    print("1. Predict a single packet")
    print("2. Predict from a CSV file")
    print("3. Exit")
    choice = input("Enter your choice (1/2/3): ")

    if choice == '1':
        protocol_type = input("Enter protocol type: ")
        service = input("Enter service: ")
        flag = input("Enter flag: ")
        print(f"Enter the following in this format:\n {' '.join(features)}")
        sample_input = input()
        sample = [float(x) for x in sample_input.split()]
        result = predict(protocol_type, service, flag, sample)
        if result == 1:
            print("Anomaly detected")
        elif result == 0:
            print("Normal packet")
        else:
            print("No model found for this combination")

    elif choice == '2':
        csv_path = input("Enter the path for the CSV file: ")
        new_data = pd.read_csv(csv_path)
        new_data['class'] = new_data.apply(
            lambda row: "Anomaly" if predict(row['protocol_type'], row['service'], row['flag'], row[features].tolist()) == 1 else "Normal", axis=1)
        new_data.to_csv('predicted_data.csv', index=False)
        print("Predictions saved to 'predicted_data.csv'")

    elif choice == '3':
        print("Exiting...")
        break

    else:
        print("Invalid choice. Please enter 1, 2, or 3.")