from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open('../models/trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['ecg_data']
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)
    return jsonify({'prediction': int(prediction[0][0] > 0.5)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
