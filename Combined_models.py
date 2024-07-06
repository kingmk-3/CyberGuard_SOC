#This prints accuracy of different models, here we modify the functioning by following label encoding
#This was basically used to check the performance of different models
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFE
import numpy as np

path = input("Enter the path for training data: ") 
data = pd.read_csv(path)
print("fetching the data...")
# This time following label encoding rather than different models for different tuples
label_encoders = {}
categorical_features = ['protocol_type', 'service', 'flag']
for feature in categorical_features:
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature])
    label_encoders[feature] = le

target = 'class'
data[target] = data[target].apply(lambda x: 1 if x != 'normal' else 0)
data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)
# We will further shorten the numebr of features usinf REF
features = ['protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'duration', 'logged_in', 'count', 'srv_count',
            'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate']
X_train, y_train = data_train[features], data_train[target]
X_test, y_test = data_test[features], data_test[target]
print("done...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Reducing the number of features...")
# To further reduce the number of coloumns
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=10)
X_train_selected = rfe.fit_transform(X_train_scaled, y_train)
X_test_selected = rfe.transform(X_test_scaled)
print("done...")
trained_models = {
    'Neural Network': MLPClassifier(hidden_layer_sizes=(13, 6), max_iter=5000, random_state=42),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'K-Means Clustering': KMeans(n_clusters=2, random_state=42),
    'SVM': SVC(kernel='linear', random_state=42),
}
print("lets train all our models...")
#we use cross_validation method to calcuate the accuracy
for model_name, model in trained_models.items():
    if model_name == 'K-Means Clustering':
        model.fit(X_train_selected)
    else:
        cross_val_scores = cross_val_score(model, X_train_selected, y_train, cv=5)
        print(f'{model_name} Cross-Validation Accuracy: {np.mean(cross_val_scores) * 100:.2f}%')
        model.fit(X_train_selected, y_train)
print("done...")
# Evaluation of models
evaluation_metrics = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': [],
}
for model_name, model in trained_models.items():
    if model_name == 'K-Means Clustering':
        y_pred = model.predict(X_test_selected)
        y_pred = [1 if pred == 1 else 0 for pred in y_pred]
    else:
        y_pred = model.predict(X_test_selected)
    
    evaluation_metrics['Model'].append(model_name)
    evaluation_metrics['Accuracy'].append(accuracy_score(y_test, y_pred))
    evaluation_metrics['Precision'].append(precision_score(y_test, y_pred))
    evaluation_metrics['Recall'].append(recall_score(y_test, y_pred))
    evaluation_metrics['F1 Score'].append(f1_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print(f'{model_name} Confusion Matrix:\n{cm}\n')

evaluation_df = pd.DataFrame(evaluation_metrics)
print(evaluation_df)

# def predict_anomaly(model_type, sample):
#     sample_scaled = scaler.transform([sample])
#     sample_selected = rfe.transform(sample_scaled)
    
#     model = trained_models.get(model_type)
#     if model:
#         sample_df = pd.DataFrame([sample], columns=features)
#         sample_scaled = scaler.transform(sample_df)
#         prediction = model.predict(sample_selected)
#         return prediction[0] 
#     else:
#         return -1