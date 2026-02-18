
# --- Imports ---
from flask import Flask, request, jsonify
from pyngrok import ngrok
import joblib
import pandas as pd
import threading
import requests
import time
from google.colab import userdata # Import userdata

# --- Upload your model if not already ---
# from google.colab import files
# uploaded = files.upload() # Uncomment and run this cell separately if you haven't uploaded the model file


# --- Load the model ---
try:
    model = joblib.load("best_model.joblib")
    print(" Model loaded successfully!")
except FileNotFoundError:
    print(" Error: Model file 'best_model.joblib' not found. Please upload the model file.")
    # You might want to stop execution here if the model is essential


# --- Initialize Flask app ---
app = Flask(__name__)

# --- Health check route ---
@app.route("/")
def home():
    return "Model is alive and ready for predictions!"

# --- Prediction route ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not isinstance(data, list):
            return jsonify({"error": "Input data must be a list of dictionaries"}), 400

        input_df = pd.DataFrame(data)

        # Ensure the input DataFrame has the same columns as the training data used for the model
        # and in the same order. Fill missing columns with 0 or a suitable default.
        # Assuming the model has a 'feature_names_in_' attribute from sklearn/xgboost
        if hasattr(model, 'feature_names_in_'):
            expected_columns = model.feature_names_in_
        else:
             # Fallback if feature_names_in_ is not available
             # This requires knowing the exact columns and their order
             # Assuming X_train is available from earlier cells
             # Make sure X_train is defined and accessible in this scope if using this fallback
             try:
                 expected_columns = X_train.columns.tolist()
             except NameError:
                 return jsonify({"error": "X_train is not defined. Cannot determine expected columns."}), 500


        input_df = input_df.reindex(columns=expected_columns, fill_value=0)


        predictions = model.predict(input_df)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Set ngrok authtoken and create tunnel ---
# Get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
# Add it to Colab Secrets (left panel, 🔑) with the name 'NGROK_AUTH_TOKEN'
try:
    ngrok_auth_token = userdata.get('32rZj66rpE2NML9SlHfNEuv8Xmi_6tZwJrwHG4PdEni8MBQSD')
    if ngrok_auth_token:
        ngrok.set_auth_token(ngrok_auth_token)
        print(" ngrok authtoken set.")
    else:
        print(" NGROK_AUTH_TOKEN not found in Colab Secrets. ngrok may not work.")

except Exception as e:
    print(f" Error retrieving ngrok authtoken: {e}")

# --- Kill existing tunnels ---
try:
    ngrok.kill()
except:
    pass

# --- Open ngrok tunnel ---
# Use a different port if 5000 is already in use
try:
    public_url_object = ngrok.connect(5000)
    public_url = public_url_object.public_url # Extract the URL string
    print(" Public URL:", public_url)
except Exception as e:
    print(f" Error creating ngrok tunnel: {e}")
    public_url = None # Set public_url to None if tunnel creation fails


# --- Run Flask in a separate thread ---
def run_flask():
    # Use threaded=True for handling multiple requests
    app.run(port=5000, threaded=True, use_reloader=False)

# Start the Flask thread only if a public_url was successfully created
if public_url:
    print("Starting Flask app in a separate thread...")
    thread = threading.Thread(target=run_flask)
    thread.daemon = True # Allow the main thread to exit even if the flask thread is running
    thread.start()
    print("Flask app thread started.")

    # --- Wait a moment for the server to start ---
    time.sleep(5) # Increased sleep time to allow server to start

    # --- Send a test POST request to /predict ---

    try:
        if public_url:
            url = f"{public_url}/predict"

            # Example input row - make sure the columns match your model's expected features
            # Create a dictionary with all expected features and default values (e.g., 0)
            # You need to know the expected feature names from your model training
            # For example, if X_train was used for training:
            try:
                example_input_data = [{col: 0.0 for col in X_train.columns}] # Use X_train columns
                print("Sending test prediction request...")
                response = requests.post(url, json=example_input_data)
                print(" Test Prediction Result:", response.json())
            except NameError:
                 print(" X_train is not defined. Skipping test prediction request.")
            except requests.exceptions.ConnectionError as ce:
                 print(f" Test Prediction Failed: Could not connect to {url}. Make sure the Flask app started correctly. Error: {ce}")
            except Exception as e:
                print(f" An unexpected error occurred during the test prediction: {e}")
        else:
            print("Skipping test prediction as ngrok tunnel was not created.")

    except Exception as e:
        print(f" An error occurred during the test request setup: {e}")

else:
    print("Flask app not started because ngrok tunnel creation failed.")
