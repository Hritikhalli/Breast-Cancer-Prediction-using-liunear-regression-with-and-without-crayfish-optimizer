import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

# Load the breast cancer dataset
df = pd.read_csv('breast-cancer.csv')

# Preprocessing
# Encode 'M' as 0 and 'B' as 1 in the 'diagnosis' column
df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})

# Split data into features and target
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model performance
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

print("Training R-squared: ", round(r2_score(y_train, y_train_pred), 2))
print("Validation R-squared: ", round(r2_score(y_val, y_val_pred), 2))
print("Test R-squared: ", round(r2_score(y_test, y_test_pred), 2))

train_accuracy = accuracy_score(y_train, np.round(y_train_pred))
val_accuracy = accuracy_score(y_val, np.round(y_val_pred))
test_accuracy = accuracy_score(y_test, np.round(y_test_pred))

train_precision = precision_score(y_train, np.round(y_train_pred), average='micro')
val_precision = precision_score(y_val, np.round(y_val_pred), average='micro')
test_precision = precision_score(y_test, np.round(y_test_pred), average='micro')


train_recall = recall_score(y_train, np.round(y_train_pred), average='micro')
val_recall = recall_score(y_val, np.round(y_val_pred), average='micro')
test_recall = recall_score(y_test, np.round(y_test_pred), average='micro')

train_f1_score = f1_score(y_train, np.round(y_train_pred), average='micro')
val_f1_score = f1_score(y_val, np.round(y_val_pred), average='micro')
test_f1_score = f1_score(y_test, np.round(y_test_pred), average='micro')

train_mcc = matthews_corrcoef(y_train, np.round(y_train_pred))
val_mcc = matthews_corrcoef(y_val, np.round(y_val_pred))
test_mcc = matthews_corrcoef(y_test, np.round(y_test_pred))

# Print metrics
print("Training Metrics:")
print("Accuracy:", train_accuracy)
print("Precision:", train_precision)
print("Recall:", train_recall)
print("F1-Score:", train_f1_score)
print("MCC:", train_mcc)

print("\nValidation Metrics:")
print("Accuracy:", val_accuracy)
print("Precision:", val_precision)
print("Recall:", val_recall)
print("F1-Score:", val_f1_score)
print("MCC:", val_mcc)

print("\nTest Metrics:")
print("Accuracy:", test_accuracy)
print("Precision:", test_precision)
print("Recall:", test_recall)
print("F1-Score:", test_f1_score)
print("MCC:", test_mcc)
