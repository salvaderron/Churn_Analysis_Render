# app.py

from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.cluster import KMeans

# Load the trained model
model_path = 'kmeans.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Verify the model type
if not isinstance(model, KMeans):
    raise TypeError("The loaded model is not an instance of KMeans")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    int_features = [int(x) for x in request.form.values()]
    # Convert the input features into a numpy array with the correct shape
    final_features = np.array(int_features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(final_features)
    
    # Determine output based on prediction
    if prediction == 0:
        output = "Customer is careless, no attention required."
    elif prediction == 1:
        output = "Customer is standard, will buy few products."
    elif prediction == 2:
        output = "Customer is Target, more attention required."
    elif prediction == 3:
        output = "Customer is careful."
    else:
        output = "Customer is sensible."
    
    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
