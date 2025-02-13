import warnings
import pandas as pd
import numpy as np
import tenseal as ts
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif

warnings.filterwarnings('ignore')

def sigmoid_approximation(z):
    return np.array([0.5 + 0.125 * val for val in z])

def initialize_weights(n_features):
    weights = np.zeros(n_features)
    bias = 0
    return weights, bias
def train_logistic_regression(X, y, learning_rate=0.01, iterations=10):
    n_features = X.shape[1]  # Get the number of features dynamically
    print(f"Training Logistic Regression with {n_features} features.")  # Debugging output
    weights, bias = initialize_weights(n_features)

    for i in range(iterations):
        linear_model = np.dot(X, weights) + bias
        y_pred = sigmoid_approximation(linear_model)

        dw = (1 / len(y)) * np.dot(X.T, (y_pred - y))
        db = (1 / len(y)) * np.sum(y_pred - y)

        weights -= learning_rate * dw
        bias -= learning_rate * db

        cost = (-1 / len(y)) * np.sum(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))
        print(f"Iteration {i}: Cost {cost}")

    return weights, bias


def predict_encrypted(X_enc, weights, bias):
    predictions = []
    for x_enc in X_enc:
        linear_model_enc = x_enc.dot(weights) + bias
        decrypted_values = linear_model_enc.decrypt()
        y_pred_enc = sigmoid_approximation(decrypted_values)

        predictions.append(1 if y_pred_enc[0] > 0.5 else 0)
    return predictions

def load_and_preprocess_data(file_path, k_features=10, fixed_selected_features=None, is_train=True):
    df = pd.read_csv(file_path).head(5000)

    # Define features
    static_features = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance']
    rating_features = [
        'Inflight wifi service', 'Ease of Online booking',
        'Online boarding', 'Seat comfort', 'Inflight entertainment',
        'On-board service', 'Leg room service', 'Baggage handling',
        'Inflight service', 'Cleanliness'
    ]

    # Combine all potential features
    features = static_features + rating_features
    X = df[features]
    y = df['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)

    # Encode categorical features
    categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Feature selection for rating features only
    if is_train:
        selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
        X_selected_rating = selector.fit_transform(X_scaled[:, len(static_features):], y)
        selected_rating_features = [rating_features[i] for i in selector.get_support(indices=True)]
    else:
        df_rating = pd.DataFrame(X_scaled[:, len(static_features):], columns=rating_features)
        X_selected_rating = df_rating[fixed_selected_features].values
        selected_rating_features = fixed_selected_features

    # Combine static features with selected rating features
    X_selected = np.hstack((X_scaled[:, :len(static_features)], X_selected_rating))  # Ensure static features are included

    selected_features = static_features + selected_rating_features  # Keep track of selected features

    print(f"Final Feature Count: {X_selected.shape[1]}")  # Debugging: Should print 16

    return X_selected, y.values, scaler, label_encoders, selected_rating_features, static_features, selected_features


def encrypt_data(context, data):
    return [ts.ckks_vector(context, row.tolist()) for row in data]

def save_model(weights, bias, scaler, label_encoders, selected_rating_features, static_features, filename="model.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump({
            'weights': weights,
            'bias': bias,
            'scaler': scaler,
            'label_encoders': label_encoders,
            'selected_rating_features': selected_rating_features,
            'static_features': static_features
        }, f)
    print(f"Model saved as {filename}")

def main():
    # Load and preprocess training data
    X_train, y_train, scaler, label_encoders, selected_rating_features, static_features, selected_features = load_and_preprocess_data(
        'C:/Users/priya/OneDrive/Documents/sem 8/DPSA LAB/CAT 1/data/train.csv', k_features=10, is_train=True
    )

    # Train logistic regression model
    weights, bias = train_logistic_regression(X_train, y_train)

    # Save the model
    save_model(weights, bias, scaler, label_encoders, selected_rating_features, static_features)

    # Load and preprocess test data using the SAME selected features
    X_test, y_test, _, _, _, _, _ = load_and_preprocess_data(
        'C:/Users/priya/OneDrive/Documents/sem 8/DPSA LAB/CAT 1/data/test.csv',
        k_features=10,
        fixed_selected_features=selected_rating_features,
        is_train=False
    )

    print("Preprocessed Test Data")

    # Initialize TenSEAL context
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = 2**40
    context.generate_galois_keys()
    context.generate_relin_keys()

    # Encrypt the test data
    X_test_encrypted = encrypt_data(context, X_test)

    # Predict the results
    y_pred_encrypted = predict_encrypted(X_test_encrypted, weights, bias)

    # Output results
    print("Accuracy:", accuracy_score(y_test, y_pred_encrypted))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_encrypted))

if __name__ == "__main__":
    main()
