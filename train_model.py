import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np

# Load the dataset
data = pd.read_csv('hand_signs_data.csv')

# Function for data augmentation (for coordinate data)
def augment_data(X, y):
    augmented_X = []
    augmented_y = []

    # Data augmentation: Adding noise, flipping, and slight rotations
    for i in range(X.shape[0]):
        # Original sample
        augmented_X.append(X[i])
        augmented_y.append(y[i])

        # Adding noise (random Gaussian noise)
        noise = np.random.normal(0, 0.1, X[i].shape)
        augmented_X.append(X[i] + noise)
        augmented_y.append(y[i])

        # Flipping (if applicable, depending on your sign language)
        flipped = X[i].copy()  # Make a copy to avoid modifying the original data
        flipped[0] = -flipped[0]  # Assuming x is the first coordinate
        augmented_X.append(flipped)
        augmented_y.append(y[i])

    # Convert lists to numpy arrays
    return np.array(augmented_X), np.array(augmented_y)

# Split the features and labels
X = data.iloc[:, 1:].values  # All columns except the first one for features (x, y, z coordinates)
y = data['Label'].values      # First column for labels

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Data augmentation
X_augmented, y_augmented = augment_data(X, y_encoded)

# Combine original and augmented data
X_combined = np.vstack((X, X_augmented))
y_combined = np.hstack((y_encoded, y_augmented))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Define the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model
best_rf_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the best model and label encoder
with open('best_rf_sign_language_model.pkl', 'wb') as f:
    pickle.dump(best_rf_model, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("Best model and label encoder saved.")
