# app.py
from flask import Flask, request, jsonify
from predict import predict_abs_error

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        predicted_error = predict_abs_error(input_data)
        return jsonify({'predicted_abs_error': predicted_error})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=True)
