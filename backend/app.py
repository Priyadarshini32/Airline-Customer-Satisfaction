from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS  
from werkzeug.security import generate_password_hash, check_password_hash  
import re  
import pickle
import tenseal as ts
import os
import logging
import numpy as np
import pandas as pd
import datetime 

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# Database Configuration
app.config['SECRET_KEY'] = 'PR@3215'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:priya32@localhost:5432/DPSA_CAT1'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Define User Model
class Login(db.Model):
    __tablename__ = 'Login'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(256), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)  
    email = db.Column(db.String(256), unique=True, nullable=False)

# Function to Validate Email
def is_valid_email(email):
    email_regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(email_regex, email)

# User Registration Route
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    email = data.get('email', '').strip()

    if not username or len(username) > 40:
        return jsonify({'error': 'Username is required and must be at most 40 characters'}), 400
    if not password or len(password) > 100:
        return jsonify({'error': 'Password is required and must be at most 100 characters'}), 400
    if not email or not is_valid_email(email):
        return jsonify({'error': 'A valid email is required'}), 400

    existing_user = Login.query.filter((Login.username == username) | (Login.email == email)).first()
    if existing_user:
        return jsonify({'error': 'Username or Email already registered'}), 400

    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')  
    new_user = Login(username=username, password=hashed_password, email=email)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'User registered successfully'}), 201
    
# User Login Route
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    if not username or not password:
        return jsonify({'error': 'Username and Password are required'}), 400

    user = Login.query.filter_by(username=username).first()

    if user and check_password_hash(user.password, password):  
        return jsonify({'message': 'Login successful', 'username': user.username}), 200
    else:
        return jsonify({'error': 'Invalid username or password'}), 401

# Load the saved model
def load_model():
    model_file = 'models/model.pkl'
    if not os.path.exists(model_file):
        return None
    with open(model_file, 'rb') as f:
        return pickle.load(f)

# Initialize TenSEAL context
def initialize_tenseal_context():
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = 2**40
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context

# Predict the satisfaction using encrypted data
def predict_encrypted(X_input, model, context):
    weights = model['weights']
    bias = model['bias']
    
    encrypted_X = ts.ckks_vector(context, X_input.tolist())

    linear_model_enc = encrypted_X.dot(weights) + bias
    decrypted_values = linear_model_enc.decrypt()
    y_pred_enc = [0.5 + 0.125 * val for val in decrypted_values]

    return 1 if y_pred_enc[0] > 0.5 else 0

# Endpoint to fetch selected features
@app.route('/selected_features', methods=['GET'])
def get_selected_features():
    model = load_model()
    if model:
        return jsonify({
            'static_features': model.get('static_features', []),
            'selected_rating_features': model.get('selected_rating_features', [])
        }), 200
    return jsonify({'error': 'Model loading failed'}), 500

# Function to calculate accuracy
def calculate_accuracy(prediction, true_label):
    return 1.0 if prediction == true_label else 0.0

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model = load_model()

    if model is None:
        return jsonify({'error': 'Model loading failed'}), 500

    # Extract model components
    selected_rating_features = model.get('selected_rating_features', None)
    static_features = model.get('static_features', None)
    label_encoders = model.get('label_encoders', {})
    scaler = model.get('scaler', None)
    weights = model.get('weights', None)

    if not (selected_rating_features and static_features and scaler and isinstance(weights, np.ndarray)):
        return jsonify({'error': 'Missing model components'}), 500

    # Combine all required features
    required_features = static_features + selected_rating_features

    # Prepare input dictionary
    feature_values_dict = {}
    missing_features = []

    for feature in required_features:
        feature_value = data.get(feature)
        if feature_value is not None:
            if isinstance(feature_value, (int, float)):
                feature_values_dict[feature] = float(feature_value)
            elif isinstance(feature_value, str) and feature in label_encoders:
                try:
                    feature_values_dict[feature] = label_encoders[feature].transform([feature_value])[0]
                except Exception as e:
                    logging.error(f"Label encoding failed for {feature}: {str(e)}")
        else:
            missing_features.append(feature)

    if missing_features:
        return jsonify({'error': f"Missing features: {missing_features}"}), 400

    if not feature_values_dict:
        return jsonify({'error': 'No valid features provided'}), 400

    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([feature_values_dict])

    # Extract all features (static + rating) and log their counts
    all_features_data = input_df[required_features].values
    logging.info(f"Static Features: {len(static_features)}, Selected Rating Features: {len(selected_rating_features)}")
    logging.info(f"Total Features in Data: {all_features_data.shape[1]} (Expected: 16)")

    # Check if the scaler is trained on the correct number of features
    if all_features_data.shape[1] != 16:
        return jsonify({'error': f"Feature count mismatch: Expected 16, but got {all_features_data.shape[1]}"}), 400

    # Scale all features
    try:
        all_features_scaled = scaler.transform(all_features_data)
    except ValueError as e:
        return jsonify({'error': f"Scaler transformation failed: {str(e)}"}), 400

    # Ensure that the scaled features match the model's expected size
    if all_features_scaled.shape[1] != len(weights):
        logging.error(f"Feature mismatch: Model expects {len(weights)}, got {all_features_scaled.shape[1]}")
        return jsonify({'error': 'Feature size mismatch'}), 400

    # Use scaled features for prediction
    X_final = all_features_scaled


    # Log feature shapes
    logging.info(f"Scaled input shape: {X_final.shape}")

    if X_final.shape[1] != len(weights):
        logging.error(f"Feature mismatch: Model expects {len(weights)}, got {X_final.shape[1]}")
        return jsonify({'error': 'Feature size mismatch'}), 400

    # Encrypt and predict
    context = initialize_tenseal_context()
    prediction = predict_encrypted(X_final[0], model, context)

    true_label = data.get("true_label")
    accuracy = calculate_accuracy(prediction, true_label) if true_label is not None else None

    return jsonify({
        'prediction': 'Satisfied' if prediction == 1 else 'Dissatisfied',
        'selected_features': list(feature_values_dict.keys()),
        'accuracy': accuracy if accuracy is not None else None,
        'debug_info': {
            'input_shape': input_df.shape,
            'scaled_shape': X_final.shape,
            'features_used': len(feature_values_dict),
            'feature_names': list(feature_values_dict.keys()),
            'feature_values': [float(val) for val in feature_values_dict.values()],
            'missing_features': missing_features
        }
    })

from datetime import datetime

class Prediction(db.Model):
    __tablename__ = 'Predictions'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(256), nullable=False)
    form_data = db.Column(db.JSON, nullable=False)
    prediction = db.Column(db.String(256), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)  # Default to current time



# Initialize TenSEAL context globally to avoid reinitializing on each request
context = initialize_tenseal_context()

from sklearn.preprocessing import LabelEncoder
@app.route('/store_prediction', methods=['POST'])
def store_prediction():
    data = request.get_json()
    username = data.get('username')
    form_data = data.get('form_data')
    prediction = data.get('prediction')

    if not username or not form_data or not prediction:
        return jsonify({'error': 'Missing required fields'}), 400

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Encrypt form data and handle string features with label encoding
    encrypted_form_data = {}

    for key, value in form_data.items():
        # If the value is a string, apply label encoding
        if isinstance(value, str):
            try:
                # Apply LabelEncoder to the string value and convert to numeric
                numeric_value = label_encoder.fit_transform([value])[0]
                encrypted_value = ts.ckks_vector(context, [numeric_value])  # Encrypt the encoded value
                # Store only the underlying vector data (as list of floats)
                encrypted_form_data[key] = encrypted_value.decrypt()  # Decrypt and store the vector as a list
            except Exception as e:
                return jsonify({'error': f'Error encoding string value for {key}: {e}'}), 400
        else:
            # For numeric values, directly encrypt them
            try:
                numeric_value = float(value)  # Try to convert value to float
                encrypted_value = ts.ckks_vector(context, [numeric_value])  # Encrypt the numeric value
                # Store only the underlying vector data (as list of floats)
                encrypted_form_data[key] = encrypted_value.decrypt()  # Decrypt and store the vector as a list
            except ValueError:
                # If the value cannot be converted to a float, store it as-is (or handle the error)
                encrypted_form_data[key] = value  # Store non-numeric values as they are

    # Save the encrypted data into the database, prediction remains plaintext
    new_prediction = Prediction(
        username=username,
        form_data=encrypted_form_data,  # Store encrypted form data as list of floats
        prediction=prediction  # Store prediction as plaintext
    )

    try:
        db.session.add(new_prediction)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Error storing data: {e}'}), 500

    return jsonify({'message': 'Encrypted form data stored successfully, prediction remains plaintext'}), 201


@app.route('/get_predictions', methods=['GET'])
def get_predictions():
    username = request.args.get('username')
    if not username:
        return jsonify({'error': 'Username is required'}), 400

    predictions = Prediction.query.filter_by(username=username).all()
    result = []

    for pred in predictions:
        result.append({
            'form_data': pred.form_data,  # Return encrypted form data as is
            'prediction': pred.prediction,  # Return the prediction as is
            'timestamp': pred.timestamp  # Return timestamp
        })

    return jsonify(result), 200



if __name__ == '__main__':
    app.run(debug=True)
