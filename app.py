from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', message=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get features input from form (comma-separated string)
    feature_str = request.form.get('feature')

    # Example: convert to list of floats (you can modify as per your real input)
    try:
        features = [float(x.strip()) for x in feature_str.split(',')]
    except Exception:
        return render_template('index.html', message=['Invalid input! Please enter numbers separated by commas.'])

    # Dummy prediction logic (replace with your ML model prediction)
    # For example: if sum of features > threshold => Malignant else Benign
    if sum(features) > 100:  # just a dummy condition
        prediction = 'Malignant'
    else:
        prediction = 'Benign'

    return render_template('index.html', message=[prediction])

if __name__ == '__main__':
    app.run(debug=True)
